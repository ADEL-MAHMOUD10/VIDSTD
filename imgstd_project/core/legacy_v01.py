"""
IMGSTD Legacy V01 Module.
Frozen reference implementation of IMGSTD::V01 encoder and decoder.
Do NOT modify — this file is the canonical V01 reference.
"""

import time
import base64
import zlib
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import zstandard as zstd

from .extractor import RLECodec
from .verifier import IntegrityVerifier


MAGIC_V01 = "IMGSTD::V01"


# ---------------------------------------------------------------------------
# V01 helpers
# ---------------------------------------------------------------------------

def _v01_prepare_image(input_source: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(input_source, str):
        img = cv2.imread(input_source, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image from path: {input_source}")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img.astype(np.uint8)
    elif isinstance(input_source, np.ndarray):
        img = input_source
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        return img
    else:
        raise TypeError(f"Unsupported image input type: {type(input_source)}")


# ---------------------------------------------------------------------------
# V01 Encoder (frozen)
# ---------------------------------------------------------------------------

class LegacyV01Encoder:
    """Frozen V01 encoder — do not modify."""

    MAGIC_PREFIX = MAGIC_V01

    def __init__(self, extractor=None, compression_level: int = 3):
        from .extractor import FastColorSegmenter
        # V01 extractor needs RLE masks; use a wrapper
        self._extractor = extractor or FastColorSegmenter()
        self._cctx = zstd.ZstdCompressor(level=compression_level)

    def encode(self, input_source: Union[str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        t_start = time.perf_counter()
        image = _v01_prepare_image(input_source)
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        raw_bytes = image.tobytes()
        original_sha256 = IntegrityVerifier.compute_sha256(raw_bytes)

        # Extract regions (V01 path — extractor returns tuple in V02; handle both)
        result = self._extractor.extract(image)
        if isinstance(result, tuple):
            regions, _, _ = result
        else:
            regions = result

        # Build base canvas using RLE (V01 style)
        base_canvas = np.zeros_like(image, dtype=np.uint8)
        for r in regions:
            bx, by, bw, bh = r.bbox
            # V01 used mask_rle; fallback if not present
            rle_str = getattr(r, "mask_rle", "")
            mask = RLECodec.decode(rle_str, (bh, bw))
            fill_color = r.dominant_color[:c]
            base_canvas[by:by + bh, bx:bx + bw][mask] = fill_color

        residual = image - base_canvas
        compressed_res = self._cctx.compress(residual.tobytes())
        res_b85 = base64.b85encode(compressed_res).decode("ascii")

        sem_lines = ["[SEM]", f"REGIONS:{len(regions)}"]
        for r in regions:
            bx, by, bw, bh = r.bbox
            rgb_str = ",".join(str(val) for val in r.dominant_color[:c])
            rle_str = getattr(r, "mask_rle", "")
            sem_lines.append(f"REG:{r.region_id}|CLS:{r.class_id}|BOX:{bx},{by},{bw},{bh}|RGB:{rgb_str}|RLE:{rle_str}")
        sem_lines.append("[/SEM]")
        sem_block = "\n".join(sem_lines)
        res_block = f"[RES]\n{res_b85}\n[/RES]"

        hdr_template = (
            "[HDR]\n"
            f"VER:V01\n"
            f"W:{w}\n"
            f"H:{h}\n"
            f"C:{c}\n"
            "B:8\n"
            f"SHA256:{original_sha256}\n"
            "SEM_OFFSET:{sem_offset:010d}\n"
            "SEM_LEN:{sem_len:010d}\n"
            "RES_OFFSET:{res_offset:010d}\n"
            "RES_LEN:{res_len:010d}\n"
            "[/HDR]"
        )
        dummy_hdr = hdr_template.format(sem_offset=0, sem_len=0, res_offset=0, res_len=0)
        magic_with_nl = f"{self.MAGIC_PREFIX}\n"
        hdr_len = len(dummy_hdr)
        sem_offset = len(magic_with_nl) + hdr_len + 1
        sem_len = len(sem_block)
        res_offset = sem_offset + sem_len + 1
        res_len = len(res_block)

        final_hdr = hdr_template.format(
            sem_offset=sem_offset, sem_len=sem_len,
            res_offset=res_offset, res_len=res_len
        )
        partial_stream = f"{magic_with_nl}{final_hdr}\n{sem_block}\n{res_block}"
        crc32_val = f"{zlib.crc32(partial_stream.encode('ascii')) & 0xFFFFFFFF:08x}"
        end_block = f"[END]\nCRC32:{crc32_val}\nTOTAL_BYTES:{len(partial_stream):010d}\n[/END]"
        imgstd_code = f"{partial_stream}\n{end_block}\n"

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        return imgstd_code, {
            "width": w, "height": h, "channels": c,
            "sha256": original_sha256,
            "region_count": len(regions),
            "total_encode_time_ms": round(total_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# V01 Decoder (frozen)
# ---------------------------------------------------------------------------

class LegacyV01Decoder:
    """Frozen V01 decoder — do not modify."""

    def __init__(self):
        self._dctx = zstd.ZstdDecompressor()

    def decode(self, stream: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()
        hdr_start = stream.find("[HDR]")
        hdr_end = stream.find("[/HDR]")
        hdr_lines = stream[hdr_start + 5:hdr_end].strip().split("\n")
        hdr_map = {}
        for line in hdr_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                hdr_map[k.strip()] = v.strip()

        w, h, c = int(hdr_map["W"]), int(hdr_map["H"]), int(hdr_map["C"])
        expected_sha = hdr_map["SHA256"]

        sem_block = ""
        if "SEM_OFFSET" in hdr_map and "SEM_LEN" in hdr_map:
            so, sl = int(hdr_map["SEM_OFFSET"]), int(hdr_map["SEM_LEN"])
            cand = stream[so:so + sl]
            if cand.startswith("[SEM]"):
                sem_block = cand
        if not sem_block:
            ss, se = stream.find("[SEM]"), stream.find("[/SEM]")
            if ss != -1 and se != -1:
                sem_block = stream[ss:se + 6]

        res_block = ""
        if "RES_OFFSET" in hdr_map and "RES_LEN" in hdr_map:
            ro, rl = int(hdr_map["RES_OFFSET"]), int(hdr_map["RES_LEN"])
            cand = stream[ro:ro + rl]
            if cand.startswith("[RES]"):
                res_block = cand
        if not res_block:
            rs, re = stream.find("[RES]"), stream.find("[/RES]")
            if rs != -1 and re != -1:
                res_block = stream[rs:re + 6]

        base_canvas = np.zeros((h, w, c), dtype=np.uint8)
        for line in sem_block.strip().split("\n"):
            line = line.strip()
            if not line.startswith("REG:"):
                continue
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

        res_content = res_block.replace("[RES]", "").replace("[/RES]", "").strip()
        compressed_bytes = base64.b85decode(res_content.encode("ascii"))
        residual_raw = self._dctx.decompress(compressed_bytes)
        residual = np.frombuffer(residual_raw, dtype=np.uint8).reshape((h, w, c))
        reconstructed = base_canvas + residual

        recon_sha = IntegrityVerifier.compute_sha256(reconstructed)
        if recon_sha != expected_sha:
            from .decoder import IntegrityError
            raise IntegrityError(f"V01 integrity failed! Expected {expected_sha}, got {recon_sha}")

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        return reconstructed, {
            "verified": True,
            "sha256": recon_sha,
            "total_decode_time_ms": round(total_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Module-level convenience helpers (mirrors encode_image / decode_imgstd API)
# ---------------------------------------------------------------------------

def encode_image_v01(
    input_source,
    output_path=None,
    compression_level: int = 3,
):
    """Encode using the frozen V01 encoder. Returns (imgstd_code, metrics)."""
    encoder = LegacyV01Encoder(compression_level=compression_level)
    code, metrics = encoder.encode(input_source)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        metrics["output_path"] = output_path
    return code, metrics


def decode_imgstd_v01(input_source: str):
    """Decode a V01 stream. Returns (image_array, metrics)."""
    decoder = LegacyV01Decoder()
    return decoder.decode(input_source)
