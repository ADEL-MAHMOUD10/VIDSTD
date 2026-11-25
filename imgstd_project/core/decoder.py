"""
IMGSTD Decoder Module — V01/V02.
Reconstructs 100% bit-exact original image from an IMGSTD::V01 or IMGSTD::V02 stream.

V02 upgrades:
  - Detects version from magic prefix
  - Reconstructs base canvas from LABEL_MAP_Z85 (uint8 label map) + color LUT
  - Reverse spatial prediction: cumsum axis=1 (mod 256) then reverse YCoCg-R
  - Falls back to V01 logic for V01 streams
"""

import os
import time
import base64
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import zstandard as zstd

from .extractor import RLECodec
from .verifier import IntegrityVerifier
from .fast_b85 import b85decode


class IntegrityError(Exception):
    """Raised when cryptographic or pixel integrity verification fails."""
    pass


class IMGSTDDecoder:
    """
    Semantic-Lossless Image Decoder supporting both V01 and V02 streams.
    Auto-detects version from magic prefix and dispatches accordingly.
    """

    def __init__(self):
        self._dctx = zstd.ZstdDecompressor()

    def _read_stream(self, input_source: str) -> str:
        """Reads stream from string or file path safely."""
        if input_source.startswith("IMGSTD::"):
            return input_source
        if len(input_source) < 4096 and os.path.exists(input_source):
            with open(input_source, "r", encoding="utf-8") as f:
                return f.read()
        return input_source

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def decode(
        self,
        input_source: str,
        output_image_path: Optional[str] = None,
        verify_sha: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reconstructs the 100% bit-exact image from an IMGSTD stream (V01 or V02).

        Returns:
            image (np.ndarray): The restored uint8 image matrix (H, W, C).
            metrics (dict): Performance, verification status, and audit report.
        """
        t_start = time.perf_counter()
        stream = self._read_stream(input_source)

        if not stream.startswith("IMGSTD::"):
            raise ValueError("Invalid stream: missing IMGSTD magic prefix.")

        if stream.startswith("IMGSTD::V04") or stream.startswith("IMGSTD::V03") or stream.startswith("IMGSTD::V02"):
            reconstructed, metrics = self._decode_v02(stream, t_start, verify_sha=verify_sha)
        else:
            # V01 or any older format
            reconstructed, metrics = self._decode_v01(stream, t_start, verify_sha=verify_sha)

        # Save image if output path specified
        if output_image_path:
            c = reconstructed.shape[2] if len(reconstructed.shape) > 2 else 1
            save_img = reconstructed
            if c == 3:
                save_img = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)
            elif c == 4:
                save_img = cv2.cvtColor(reconstructed, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(output_image_path, save_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            metrics["output_path"] = output_image_path

        return reconstructed, metrics

    # ------------------------------------------------------------------
    # V02 decoder
    # ------------------------------------------------------------------

    def _decode_v02(self, stream: str, t_start: float, verify_sha: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decodes an IMGSTD::V02 stream."""

        # 1. Parse [HDR]
        hdr_map = self._parse_hdr(stream)
        w = int(hdr_map["W"])
        h = int(hdr_map["H"])
        c = int(hdr_map["C"])
        expected_sha = hdr_map["SHA256"]

        # 2. Extract [SEM] block
        sem_block = self._extract_block(stream, hdr_map, "SEM", "[SEM]", "[/SEM]")

        # 3. Parse label map from LABEL_MAP_Z85 line in [SEM]
        t_base_start = time.perf_counter()
        label_map_b85 = None
        color_lut = np.zeros((256, c), dtype=np.uint8)
        region_count = 0

        for line in sem_block.strip().split("\n"):
            line = line.strip()
            if line.startswith("LABEL_MAP_Z85:"):
                label_map_b85 = line[len("LABEL_MAP_Z85:"):]
            elif line.startswith("REG:"):
                region_count += 1
                parts = line.split("|")
                fd = {}
                for p in parts:
                    if ":" in p:
                        fk, fv = p.split(":", 1)
                        fd[fk] = fv
                label_id = int(fd["REG"])  # region_id == label_id (1-indexed)
                rgb = [int(v) for v in fd["RGB"].split(",")]
                color_lut[label_id] = rgb[:c]

        if label_map_b85 is None:
            raise ValueError("Malformed V02 stream: missing LABEL_MAP_Z85 in [SEM].")

        # Decompress label map
        label_map_bytes = self._dctx.decompress(b85decode(label_map_b85))
        label_map = np.frombuffer(label_map_bytes, dtype=np.uint8).reshape(h, w)

        # Build base canvas via LUT
        base_canvas = color_lut[label_map]  # (H, W, c)
        t_base_ms = (time.perf_counter() - t_base_start) * 1000.0

        # 4. Extract and decompress [RES]
        t_res_start = time.perf_counter()
        res_block = self._extract_block(stream, hdr_map, "RES", "[RES]", "[/RES]")
        res_content = res_block.replace("[RES]", "").replace("[/RES]", "").strip()
        compressed_bytes = b85decode(res_content)
        delta_raw = self._dctx.decompress(compressed_bytes)
        delta = np.frombuffer(delta_raw, dtype=np.uint8).reshape(h, w, c)
        t_res_ms = (time.perf_counter() - t_res_start) * 1000.0

        # 5. Reverse horizontal delta (ultra-fast in-place uint8 accumulator)
        pred = np.add.accumulate(delta, axis=1, dtype=np.uint8)

        # 6. Reverse YCoCg-R (3-channel only, fully vectorized in uint8)
        if c == 3:
            G = pred[:, :, 0]
            Co = pred[:, :, 1]
            Cg = pred[:, :, 2]
            residual = np.empty_like(pred)
            residual[:, :, 0] = G + Co
            residual[:, :, 1] = G
            residual[:, :, 2] = G + Cg
        else:
            residual = pred

        # 7. Reconstruct image
        reconstructed = base_canvas + residual  # uint8 wrap

        # 8. Cryptographic verification
        recon_sha = IntegrityVerifier.compute_sha256(reconstructed)
        sha_match = (recon_sha == expected_sha)
        if verify_sha and not sha_match:
            raise IntegrityError(
                f"V02 integrity failed! Expected SHA: {expected_sha}, Reconstructed SHA: {recon_sha}"
            )

        if stream.startswith("IMGSTD::V04"):
            protocol = "IMGSTD::V04"
        elif stream.startswith("IMGSTD::V03"):
            protocol = "IMGSTD::V03"
        else:
            protocol = "IMGSTD::V02"
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        metrics = {
            "verified": sha_match,
            "protocol": protocol,
            "sha256": recon_sha,
            "expected_sha256": expected_sha,
            "sha256_match": sha_match,
            "mse": 0.0 if sha_match else None,
            "psnr_db": float("inf") if sha_match else None,
            "diff_pixel_count": 0 if sha_match else None,
            "fidelity_percentage": 100.0 if sha_match else None,
            "edited": not sha_match,
            "dimensions": {"width": w, "height": h, "channels": c},
            "region_count": region_count,
            "total_decode_time_ms": round(total_time_ms, 2),
            "base_reconstruct_time_ms": round(t_base_ms, 2),
            "residual_decompress_time_ms": round(t_res_ms, 2),
        }
        return reconstructed, metrics

    # ------------------------------------------------------------------
    # V01 decoder (fallback)
    # ------------------------------------------------------------------

    def _decode_v01(self, stream: str, t_start: float, verify_sha: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decodes an IMGSTD::V01 stream."""

        hdr_map = self._parse_hdr(stream)
        w = int(hdr_map["W"])
        h = int(hdr_map["H"])
        c = int(hdr_map["C"])
        expected_sha = hdr_map["SHA256"]

        sem_block = self._extract_block(stream, hdr_map, "SEM", "[SEM]", "[/SEM]")
        res_block = self._extract_block(stream, hdr_map, "RES", "[RES]", "[/RES]")

        # Reconstruct base canvas (V01: RLE masks)
        t_base_start = time.perf_counter()
        base_canvas = np.zeros((h, w, c), dtype=np.uint8)
        region_count = 0

        for line in sem_block.strip().split("\n"):
            line = line.strip()
            if not line.startswith("REG:"):
                continue
            region_count += 1
            parts = line.split("|")
            fd = {}
            for p in parts:
                if ":" in p:
                    fk, fv = p.split(":", 1)
                    fd[fk] = fv
            bx, by, bw, bh = [int(v) for v in fd["BOX"].split(",")]
            rgb = [int(v) for v in fd["RGB"].split(",")]
            rle_str = fd.get("RLE", "")
            mask = RLECodec.decode(rle_str, (bh, bw))
            base_canvas[by:by + bh, bx:bx + bw][mask] = rgb[:c]

        t_base_ms = (time.perf_counter() - t_base_start) * 1000.0

        # Decompress residual
        t_res_start = time.perf_counter()
        res_content = res_block.replace("[RES]", "").replace("[/RES]", "").strip()
        compressed_bytes = base64.b85decode(res_content.encode("ascii"))
        residual_raw = self._dctx.decompress(compressed_bytes)
        residual = np.frombuffer(residual_raw, dtype=np.uint8).reshape((h, w, c))
        t_res_ms = (time.perf_counter() - t_res_start) * 1000.0

        reconstructed = base_canvas + residual

        recon_sha = IntegrityVerifier.compute_sha256(reconstructed)
        sha_match = (recon_sha == expected_sha)
        if verify_sha and not sha_match:
            raise IntegrityError(
                f"V01 integrity failed! Expected SHA: {expected_sha}, Reconstructed SHA: {recon_sha}"
            )

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        metrics = {
            "verified": sha_match,
            "protocol": "IMGSTD::V01",
            "sha256": recon_sha,
            "expected_sha256": expected_sha,
            "sha256_match": sha_match,
            "mse": 0.0 if sha_match else None,
            "psnr_db": float("inf") if sha_match else None,
            "diff_pixel_count": 0 if sha_match else None,
            "fidelity_percentage": 100.0 if sha_match else None,
            "edited": not sha_match,
            "dimensions": {"width": w, "height": h, "channels": c},
            "region_count": region_count,
            "total_decode_time_ms": round(total_time_ms, 2),
            "base_reconstruct_time_ms": round(t_base_ms, 2),
            "residual_decompress_time_ms": round(t_res_ms, 2),
        }
        return reconstructed, metrics

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _parse_hdr(self, stream: str) -> Dict[str, str]:
        """Parses [HDR] block into a key-value dict."""
        hdr_start = stream.find("[HDR]")
        hdr_end = stream.find("[/HDR]")
        if hdr_start == -1 or hdr_end == -1:
            raise ValueError("Malformed stream: missing [HDR] block.")
        hdr_lines = stream[hdr_start + 5:hdr_end].strip().split("\n")
        hdr_map: Dict[str, str] = {}
        for line in hdr_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                hdr_map[k.strip()] = v.strip()
        return hdr_map

    def _extract_block(
        self, stream: str, hdr_map: Dict[str, str],
        block_name: str, open_tag: str, close_tag: str
    ) -> str:
        """Extracts a named block using exact UTF-8 byte offsets (fast path) or search fallback."""
        offset_key = f"{block_name}_OFFSET"
        len_key = f"{block_name}_LEN"
        if offset_key in hdr_map and len_key in hdr_map:
            off = int(hdr_map[offset_key])
            ln = int(hdr_map[len_key])
            stream_bytes = stream.encode("utf-8")
            if 0 <= off and off + ln <= len(stream_bytes):
                candidate = stream_bytes[off:off + ln].decode("utf-8", errors="replace")
                if candidate.startswith(open_tag):
                    return candidate

        # Fallback: search
        s = stream.find(open_tag)
        e = stream.find(close_tag)
        if s != -1 and e != -1:
            return stream[s:e + len(close_tag)]
        raise ValueError(f"Malformed stream: missing {open_tag} block.")


def decode_imgstd(
    input_source: str,
    output_image_path: Optional[str] = None,
    verify_sha: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Helper function to decode an IMGSTD stream (V01 or V02) and optionally save to disk."""
    decoder = IMGSTDDecoder()
    return decoder.decode(input_source, output_image_path=output_image_path, verify_sha=verify_sha)
