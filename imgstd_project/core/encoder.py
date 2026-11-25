"""
IMGSTD Encoder Module — V03.
Converts any digital image into a structured, deterministic Semantic-Lossless IMGSTD::V03 stream.

V03 upgrades:
  - YOLO11n-seg Instance Segmentation + Honest Background Modeling (0900-0999)
  - Region line format: REG:001|KIND:OBJ|CLS:0003|CONF:0.94|BOX:...|RGB:...|COV:...
  - uint8 Label Map (H×W) with Base85 Zstandard compression
  - YCoCg-R color decorrelation + horizontal delta filter on residual
  - Multi-threaded ZstdCompressor(level=3, threads=-1)
  - [HDR] fields: VER:V03, PRED:YCOCG_HDELTA, LABELMAP_OFFSET, LABELMAP_LEN
"""

import time
import base64
import zlib
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import zstandard as zstd

from .extractor import BaseSemanticExtractor, YoloSemanticExtractor, YoloWorldSemanticExtractor, FastColorSegmenter
from .embedding import RegionEmbeddingGenerator
from .vlm_captioner import get_vlm_captioner
from .verifier import IntegrityVerifier
from .fast_b85 import b85encode


class IMGSTDEncoder:
    """
    Standard Semantic-Lossless Image Encoder (V03 & V02).
    Encodes image matrices into hierarchical [HDR], [SEM], [RES], and [END] layers.
    """

    MAGIC_PREFIX = "IMGSTD::V04"

    def __init__(
        self,
        extractor: Optional[BaseSemanticExtractor] = None,
        compression_level: int = 1,
        version: str = "V04",
        scene_model: Optional[str] = None,
        enable_smolvlm2: Optional[bool] = None,
    ):
        from .config import load_settings, normalize_scene_model_name
        cfg = load_settings()
        self.version = version.upper()
        self.MAGIC_PREFIX = f"IMGSTD::{self.version}"
        self.scene_model = normalize_scene_model_name(scene_model or cfg.get("scene_model", "mobileclip2-s0"))
        self.enable_smolvlm2 = bool(enable_smolvlm2) if enable_smolvlm2 is not None else bool(cfg.get("enable_smolvlm2", False))
        if extractor is not None:
            self.extractor = extractor
        else:
            yolo_conf = float(cfg.get("yolo_conf", 0.40))
            self.extractor = YoloWorldSemanticExtractor(conf=yolo_conf) if self.version == "V04" else (YoloSemanticExtractor(conf=yolo_conf) if self.version == "V03" else FastColorSegmenter())

        self.compression_level = compression_level
        # Multi-threaded compressor
        self._cctx = zstd.ZstdCompressor(level=self.compression_level, threads=-1)

    def _prepare_image(self, input_source: Union[str, np.ndarray]) -> np.ndarray:
        """Loads and normalizes image into a uint8 NumPy array with channels."""
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
            raise ValueError("Input source must be a valid file path or NumPy ndarray.")

    def encode(
        self,
        input_source: Union[str, np.ndarray],
        include_residuals: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes the V03/V04 encoding pipeline.
        When include_residuals=False, extracts semantics/OCR/scene context without pixel residual compression.

        Returns:
            imgstd_code (str): The complete standardized IMGSTD stream.
            metrics (dict): Performance, dimensions, and integrity seal metadata.
        """
        t_start = time.perf_counter()

        # 1. Prepare image
        image = self._prepare_image(input_source)
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        original_size = int(image.nbytes)
        original_sha256 = IntegrityVerifier.compute_sha256(image) if include_residuals else "SEMANTIC_REF"

        # 2. Extract semantic regions, label map, and text layer
        t_sem_start = time.perf_counter()
        extract_result = self.extractor.extract(image)
        if len(extract_result) == 4:
            regions, label_map, color_lut, texts = extract_result
        else:
            regions, label_map, color_lut = extract_result
            texts = getattr(self.extractor, "last_texts", [])
        t_sem_ms = (time.perf_counter() - t_sem_start) * 1000.0

        if include_residuals:
            # 3. Build base canvas via vectorized LUT lookup
            base_canvas = color_lut[label_map]

            # 4. Compute residual (uint8 wrap-around)
            residual = image - base_canvas

            # 5. YCoCg-R decorrelation on residual (3-channel only, fully vectorized in uint8)
            t_res_start = time.perf_counter()
            if c == 3:
                G = residual[:, :, 1]
                Co = residual[:, :, 0] - G
                Cg = residual[:, :, 2] - G
                decorated = np.empty_like(residual)
                decorated[:, :, 0] = G
                decorated[:, :, 1] = Co
                decorated[:, :, 2] = Cg
            else:
                decorated = residual.copy()

            # 6. Horizontal delta filter (in uint8)
            delta = decorated.copy()
            delta[:, 1:, :] -= decorated[:, :-1, :]

            # 7. Compress residual delta with multi-threaded zstd
            compressed_res = self._cctx.compress(delta.tobytes())
            res_b85 = b85encode(compressed_res)
            t_res_ms = (time.perf_counter() - t_res_start) * 1000.0

            # 8. Serialize label_map: uint8 H×W → zstd → base85
            label_map_bytes = self._cctx.compress(label_map.tobytes())
            label_map_b85 = b85encode(label_map_bytes)
        else:
            res_b85 = ""
            label_map_b85 = ""
            t_res_ms = 0.0

        # 8b. Keyframe Micro-VLM / MobileCLIP2 Scene & Region Captioning (V04)
        scene_desc = ""
        if self.version == "V04":
            try:
                vlm = get_vlm_captioner(enable_neural=self.enable_smolvlm2, scene_model=self.scene_model)
                scene_desc = vlm.caption_scene(image, regions=regions, texts=texts, scene_model=self.scene_model)
                reg_attrs = vlm.describe_regions(image, regions=regions, label_map=label_map)
                for r in regions:
                    if not getattr(r, "attributes", None) and r.region_id in reg_attrs:
                        r.attributes = reg_attrs[r.region_id]
            except Exception:
                pass

        # 9. Format [SEM] block (V03/V04: KIND, CONF, COV fields, no RLE)
        sem_lines = ["[SEM]", f"REGIONS:{len(regions)}"]
        if self.version == "V04" and scene_desc:
            esc_scene_desc = scene_desc.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('|', '\\|')
            sem_lines.append(f'SCENE_DESC:"{esc_scene_desc}"')
        sem_lines.append(f"LABEL_MAP_Z85:{label_map_b85}")

        for r in regions:
            bx, by, bw, bh = r.bbox
            rgb_str = ",".join(str(val) for val in r.dominant_color[:c])
            cov_str = f"{r.coverage_pct:.2f}%"
            kind_str = getattr(r, "kind", "OBJ")
            conf_str = f"{getattr(r, 'conf', 1.00):.2f}"
            refl_str = f"|REFL:{r.parent_structure[:4]}" if getattr(r, "is_reflection", False) and r.parent_structure else ""
            name_str = f'|NAME:"{r.class_name}"' if getattr(r, "class_name", None) else ""
            cat_str = f'|CAT:"{r.category}"' if getattr(r, "category", None) else ""
            emb_str = ""
            if getattr(r, "embedding", None) is not None:
                emb_b85 = RegionEmbeddingGenerator.encode_b85(r.embedding)
                emb_str = f"|EMB:{emb_b85}"
            attr_str = ""
            if getattr(r, "attributes", None):
                esc_attr = str(r.attributes).replace('"', '\\"').replace('\n', ' ').replace('|', '\\|')
                attr_str = f'|ATTR:"{esc_attr}"'

            sem_lines.append(
                f"REG:{r.region_id}|KIND:{kind_str}|CLS:{r.class_id}{name_str}{cat_str}|CONF:{conf_str}|BOX:{bx},{by},{bw},{bh}|RGB:{rgb_str}|COV:{cov_str}{refl_str}{emb_str}{attr_str}"
            )
        sem_lines.append("[/SEM]")
        sem_block = "\n".join(sem_lines)

        # 10. Format [TXT] block (V04)
        if self.version == "V04":
            txt_lines = ["[TXT]", f"TEXTS:{len(texts)}"]
            for idx, t in enumerate(texts, 1):
                tx, ty, tw, th = t.bbox
                esc_str = (
                    str(t.text)
                    .replace('\\', '\\\\')
                    .replace('\n', '\\n')
                    .replace('\r', '')
                    .replace('"', '\\"')
                    .replace('|', '\\|')
                )
                txt_lines.append(f'TXT:{t.text_id}|CONF:{t.confidence:.2f}|BOX:{tx},{ty},{tw},{th}|STR:"{esc_str}"')
            txt_lines.append("[/TXT]")
            txt_block = "\n".join(txt_lines)
        else:
            txt_block = ""

        # 11. Format [RES] block
        res_block = f"[RES]\n{res_b85}\n[/RES]"

        # 12. Calculate exact UTF-8 byte offsets with 10-digit fixed format
        magic_with_nl = f"{self.MAGIC_PREFIX}\n"
        magic_bytes_len = len(magic_with_nl.encode("utf-8"))
        sem_bytes = sem_block.encode("utf-8")
        res_bytes = res_block.encode("utf-8")
        label_map_line = f"LABEL_MAP_Z85:{label_map_b85}"
        label_map_line_offset_in_sem = sem_bytes.find(b"LABEL_MAP_Z85:")
        labelmap_len = len(label_map_line.encode("utf-8"))

        if self.version == "V04":
            hdr_template = (
                "[HDR]\n"
                f"VER:{self.version}\n"
                f"W:{w}\n"
                f"H:{h}\n"
                f"C:{c}\n"
                "B:8\n"
                f"PRED:YCOCG_HDELTA\n"
                f"SHA256:{original_sha256}\n"
                "SEM_OFFSET:{sem_offset:010d}\n"
                "SEM_LEN:{sem_len:010d}\n"
                "TXT_OFFSET:{txt_offset:010d}\n"
                "TXT_LEN:{txt_len:010d}\n"
                "RES_OFFSET:{res_offset:010d}\n"
                "RES_LEN:{res_len:010d}\n"
                "LABELMAP_OFFSET:{labelmap_offset:010d}\n"
                "LABELMAP_LEN:{labelmap_len:010d}\n"
                "[/HDR]"
            )
            dummy_hdr = hdr_template.format(
                sem_offset=0, sem_len=0, txt_offset=0, txt_len=0,
                res_offset=0, res_len=0, labelmap_offset=0, labelmap_len=0
            )
            hdr_len = len(dummy_hdr.encode("utf-8"))
            txt_bytes = txt_block.encode("utf-8")
            sem_offset = magic_bytes_len + hdr_len + 1
            sem_len = len(sem_bytes)
            txt_offset = sem_offset + sem_len + 1
            txt_len = len(txt_bytes)
            res_offset = txt_offset + txt_len + 1
            res_len = len(res_bytes)
            labelmap_offset = sem_offset + label_map_line_offset_in_sem

            final_hdr = hdr_template.format(
                sem_offset=sem_offset, sem_len=sem_len,
                txt_offset=txt_offset, txt_len=txt_len,
                res_offset=res_offset, res_len=res_len,
                labelmap_offset=labelmap_offset, labelmap_len=labelmap_len
            )
            partial_stream = f"{magic_with_nl}{final_hdr}\n{sem_block}\n{txt_block}\n{res_block}"
        else:
            hdr_template = (
                "[HDR]\n"
                f"VER:{self.version}\n"
                f"W:{w}\n"
                f"H:{h}\n"
                f"C:{c}\n"
                "B:8\n"
                f"PRED:YCOCG_HDELTA\n"
                f"SHA256:{original_sha256}\n"
                "SEM_OFFSET:{sem_offset:010d}\n"
                "SEM_LEN:{sem_len:010d}\n"
                "RES_OFFSET:{res_offset:010d}\n"
                "RES_LEN:{res_len:010d}\n"
                "LABELMAP_OFFSET:{labelmap_offset:010d}\n"
                "LABELMAP_LEN:{labelmap_len:010d}\n"
                "[/HDR]"
            )
            dummy_hdr = hdr_template.format(
                sem_offset=0, sem_len=0, res_offset=0, res_len=0,
                labelmap_offset=0, labelmap_len=0
            )
            hdr_len = len(dummy_hdr.encode("utf-8"))
            sem_offset = magic_bytes_len + hdr_len + 1
            sem_len = len(sem_bytes)
            res_offset = sem_offset + sem_len + 1
            res_len = len(res_bytes)
            labelmap_offset = sem_offset + label_map_line_offset_in_sem

            final_hdr = hdr_template.format(
                sem_offset=sem_offset, sem_len=sem_len,
                res_offset=res_offset, res_len=res_len,
                labelmap_offset=labelmap_offset, labelmap_len=labelmap_len
            )
            partial_stream = f"{magic_with_nl}{final_hdr}\n{sem_block}\n{res_block}"

        partial_bytes = partial_stream.encode("utf-8")
        crc32_val = f"{zlib.crc32(partial_bytes) & 0xFFFFFFFF:08x}"
        end_block = f"[END]\nCRC32:{crc32_val}\nTOTAL_BYTES:{len(partial_bytes):010d}\n[/END]"

        imgstd_code = f"{partial_stream}\n{end_block}\n"

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        compressed_size = len(imgstd_code.encode("utf-8"))
        compression_ratio = round(original_size / max(1, compressed_size), 3)

        metrics = {
            "width": w,
            "height": h,
            "channels": c,
            "sha256": original_sha256,
            "region_count": len(regions),
            "text_count": len(texts),
            "original_bytes": original_size,
            "encoded_bytes": compressed_size,
            "compression_ratio": compression_ratio,
            "total_encode_time_ms": round(total_time_ms, 2),
            "semantic_extract_time_ms": round(t_sem_ms, 2),
            "residual_compress_time_ms": round(t_res_ms, 2),
            "protocol": self.MAGIC_PREFIX,
            "scene_model": self.scene_model,
            "smolvlm2_enabled": self.enable_smolvlm2,
        }

        return imgstd_code, metrics


def encode_image(
    input_source: Union[str, np.ndarray],
    output_path: Optional[str] = None,
    extractor: Optional[BaseSemanticExtractor] = None,
    compression_level: int = 1,
    version: str = "V04",
    scene_model: Optional[str] = None,
    enable_smolvlm2: Optional[bool] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Helper function to encode an image and optionally write to file."""
    encoder = IMGSTDEncoder(
        extractor=extractor,
        compression_level=compression_level,
        version=version,
        scene_model=scene_model,
        enable_smolvlm2=enable_smolvlm2,
    )
    code, metrics = encoder.encode(input_source)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        metrics["output_path"] = output_path
    return code, metrics
