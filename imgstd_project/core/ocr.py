"""
IMGSTD Multilingual Scene OCR Module — V04.
Extracts English, Arabic, and multilingual scene text in <45 ms on CPU using RapidOCR (ONNX).
Outputs structured TextDescriptors for serialization into the [TXT] layer.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger("imgstd.ocr")


@dataclass
class TextDescriptor:
    text_id: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    text: str


_CACHED_OCR_ENGINE = None


def get_ocr_engine() -> Any:
    """Lazy singleton loader for RapidOCR engine with ONNX Runtime."""
    global _CACHED_OCR_ENGINE
    if _CACHED_OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _CACHED_OCR_ENGINE = RapidOCR()
        except Exception as exc:
            logger.warning("RapidOCR failed to initialize (%s); OCR [TXT] layer disabled.", exc)
            _CACHED_OCR_ENGINE = None
    return _CACHED_OCR_ENGINE


class FastSceneOCR:
    """
    High-speed CPU OCR Engine for IMGSTD::V04 [TXT] layer.
    Extracts text tokens with confidence scores and pixel bounding boxes.
    Supports English, Arabic, numerals, and multilingual typography.
    """

    def __init__(self, use_ocr: bool = True, min_conf: float = 0.45):
        self.use_ocr = use_ocr
        self.min_conf = min_conf

    def extract(self, image: np.ndarray) -> List[TextDescriptor]:
        """
        Extracts text instances from a uint8 image (H, W, C).
        Returns a list of TextDescriptor objects.
        """
        if not self.use_ocr:
            return []

        engine = get_ocr_engine()
        if engine is None:
            return []

        h, w = image.shape[:2]
        # For small images (< 450px), upscale 2x to detect small storefront text lines
        # For large images (> 960px), downscale to 960 for 2x faster CPU DBNet inference
        if max(h, w) < 450:
            scale = 2.0
            ocr_input = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif max(h, w) > 960:
            scale = 960.0 / max(h, w)
            ocr_input = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            ocr_input = image

        try:
            raw_res, _ = engine(ocr_input)
            if not raw_res:
                return []

            results = []
            for idx, item in enumerate(raw_res, 1):
                poly, txt, score_val = item
                score = float(score_val)
                if score < self.min_conf:
                    continue

                xs = [p[0] / scale for p in poly]
                ys = [p[1] / scale for p in poly]
                bx = max(0, int(round(min(xs))))
                by = max(0, int(round(min(ys))))
                bw = min(w - bx, max(1, int(round(max(xs) - min(xs)))))
                bh = min(h - by, max(1, int(round(max(ys) - min(ys)))))

                clean_text = str(txt).strip()
                if not clean_text:
                    continue

                results.append(TextDescriptor(
                    text_id=f"{idx:03d}",
                    confidence=round(score, 2),
                    bbox=(bx, by, bw, bh),
                    text=clean_text
                ))
            return results
        except Exception as exc:
            logger.warning("RapidOCR inference failed on image (%s).", exc)
            return []
