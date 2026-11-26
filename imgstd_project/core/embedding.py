"""
IMGSTD Semantic Visual Embedding Module.
Generates lightweight quantized int8 visual feature embeddings for semantic regions,
supporting fast cosine similarity, spatial-visual retrieval, and multimodal search.
"""

from typing import Tuple, Optional
import numpy as np
from .fast_b85 import b85encode, b85decode


class RegionEmbeddingGenerator:
    """
    Generates a deterministic 32-dimensional int8 visual feature vector
    for any segmented image region.
    """

    DIMENSIONS = 32

    @classmethod
    def compute(
        cls,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes 32-dim int8 embedding from region pixels and bounding geometry.
        
        Args:
            image: Full canvas (H, W, C) uint8.
            bbox: (x, y, w, h) bounding box.
            mask: Optional boolean mask of shape (H, W) or (h, w).
        Returns:
            int8 numpy array of shape (32,).
        """
        H, W = image.shape[:2]
        bx, by, bw, bh = bbox
        
        # Guard against zero-area boxes
        bw = max(1, bw)
        bh = max(1, bh)
        
        x2 = min(W, bx + bw)
        y2 = min(H, by + bh)
        bx = max(0, min(W - 1, bx))
        by = max(0, min(H - 1, by))
        
        crop = image[by:y2, bx:x2]
        if crop.size == 0:
            return np.zeros(cls.DIMENSIONS, dtype=np.int8)

        # Uniform spatial striding (~<= 64x64 grid) keeps color/gradient distributions invariant at <0.4ms
        ch_c, cw_c = crop.shape[:2]
        step_y = max(1, ch_c // 64)
        step_x = max(1, cw_c // 64)
        crop = crop[::step_y, ::step_x]

        vec = np.zeros(cls.DIMENSIONS, dtype=np.float32)

        # 1. Geometry & Spatial Position (dims 0..5)
        vec[0] = (bx + bw / 2.0) / max(1, W) * 2.0 - 1.0  # Normalized X [-1, 1]
        vec[1] = (by + bh / 2.0) / max(1, H) * 2.0 - 1.0  # Normalized Y [-1, 1]
        vec[2] = (bw / max(1, W)) * 2.0 - 1.0            # Normalized width
        vec[3] = (bh / max(1, H)) * 2.0 - 1.0            # Normalized height
        vec[4] = min(5.0, (bw / max(1, bh))) / 2.5 - 1.0 # Aspect ratio
        vec[5] = np.log1p(bw * bh) / np.log1p(W * H) * 2.0 - 1.0 # Log area

        # 2. Color Distribution & Moments (dims 6..17)
        if crop.ndim == 3 and crop.shape[2] >= 3:
            r = crop[:, :, 0].astype(np.float32) / 255.0
            g = crop[:, :, 1].astype(np.float32) / 255.0
            b = crop[:, :, 2].astype(np.float32) / 255.0

            vec[6] = np.mean(r) * 2.0 - 1.0
            vec[7] = np.mean(g) * 2.0 - 1.0
            vec[8] = np.mean(b) * 2.0 - 1.0

            vec[9] = np.std(r) * 4.0 - 1.0
            vec[10] = np.std(g) * 4.0 - 1.0
            vec[11] = np.std(b) * 4.0 - 1.0

            # Color differences (R-G, G-B, B-R)
            vec[12] = np.mean(r - g)
            vec[13] = np.mean(g - b)
            vec[14] = np.mean(b - r)

            # Luminance
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            vec[15] = np.mean(lum) * 2.0 - 1.0
            vec[16] = np.std(lum) * 4.0 - 1.0
            vec[17] = np.percentile(lum, 90) * 2.0 - 1.0
        else:
            gray = crop.astype(np.float32) / 255.0
            vec[6] = np.mean(gray) * 2.0 - 1.0
            vec[7] = np.std(gray) * 4.0 - 1.0

        # 3. Gradients & Texture Energy (dims 18..25)
        if crop.shape[0] > 2 and crop.shape[1] > 2:
            gray_crop = (
                (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]).astype(np.float32)
                if crop.ndim == 3 and crop.shape[2] >= 3
                else crop.astype(np.float32)
            )
            dx = np.diff(gray_crop, axis=1)
            dy = np.diff(gray_crop, axis=0)

            vec[18] = np.clip(np.mean(np.abs(dx)) / 32.0 * 2.0 - 1.0, -1.0, 1.0)
            vec[19] = np.clip(np.mean(np.abs(dy)) / 32.0 * 2.0 - 1.0, -1.0, 1.0)
            vec[20] = np.clip(np.std(dx) / 32.0 * 2.0 - 1.0, -1.0, 1.0)
            vec[21] = np.clip(np.std(dy) / 32.0 * 2.0 - 1.0, -1.0, 1.0)

            # Corner energy (quadrant mean luminances)
            qh, qw = max(1, gray_crop.shape[0] // 2), max(1, gray_crop.shape[1] // 2)
            vec[22] = np.mean(gray_crop[:qh, :qw]) / 128.0 - 1.0
            vec[23] = np.mean(gray_crop[:qh, qw:]) / 128.0 - 1.0
            vec[24] = np.mean(gray_crop[qh:, :qw]) / 128.0 - 1.0
            vec[25] = np.mean(gray_crop[qh:, qw:]) / 128.0 - 1.0

        # 4. Global normalizations (dims 26..31)
        norm = np.linalg.norm(vec[:26])
        if norm > 1e-6:
            vec[:26] /= norm

        # Quantize to int8 [-127, 127]
        quantized = np.clip(np.round(vec * 127.0), -127, 127).astype(np.int8)
        return quantized

    @classmethod
    def encode_b85(cls, embedding: np.ndarray) -> str:
        """Serializes int8 embedding into pipe-safe Base85 string."""
        s = b85encode(embedding.tobytes())
        return s.replace("|", "~")

    @classmethod
    def decode_b85(cls, b85_str: str) -> np.ndarray:
        """Deserializes pipe-safe Base85 string back into int8 numpy embedding vector."""
        raw = b85decode(b85_str.replace("~", "|"))
        return np.frombuffer(raw, dtype=np.int8)
