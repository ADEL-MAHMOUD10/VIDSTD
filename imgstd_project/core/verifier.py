"""
IMGSTD Integrity Verifier Module.
Provides mathematical exactness verification, SHA-256 checksums, and pixel audit reports.
"""

import hashlib
from typing import Dict, Any, Union
import numpy as np


class IntegrityVerifier:
    """Mathematical and cryptographic verification suite for IMGSTD."""

    @staticmethod
    def compute_sha256(data: Union[bytes, np.ndarray]) -> str:
        """Computes SHA-256 hex digest of raw bytes or numpy array buffer."""
        if isinstance(data, np.ndarray):
            raw_bytes = data.tobytes()
        else:
            raw_bytes = data
        return hashlib.sha256(raw_bytes).hexdigest()

    @classmethod
    def verify_bit_exact(cls, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
        """
        Conducts a pixel-by-pixel mathematical audit comparing original vs reconstructed images.
        Guarantees 100.000% bit-exact verification.
        """
        orig_shape = original.shape
        recon_shape = reconstructed.shape

        if orig_shape != recon_shape:
            return {
                "verified": False,
                "reason": f"Shape mismatch: original {orig_shape} vs reconstructed {recon_shape}",
                "mse": float("inf"),
                "diff_pixel_count": -1,
                "sha256_match": False,
            }

        if original.dtype != reconstructed.dtype:
            return {
                "verified": False,
                "reason": f"Dtype mismatch: original {original.dtype} vs reconstructed {reconstructed.dtype}",
                "mse": float("inf"),
                "diff_pixel_count": -1,
                "sha256_match": False,
            }

        # Cryptographic check
        orig_sha = cls.compute_sha256(original)
        recon_sha = cls.compute_sha256(reconstructed)
        sha_match = (orig_sha == recon_sha)

        # Pixel difference check
        diff_mask = (original != reconstructed)
        diff_count = int(np.sum(diff_mask))
        exact_match = (diff_count == 0)

        # Mean Squared Error & PSNR
        if exact_match:
            mse = 0.0
            psnr = float("inf")
        else:
            mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
            psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")

        is_verified = exact_match and sha_match

        return {
            "verified": is_verified,
            "sha256_match": sha_match,
            "original_sha256": orig_sha,
            "reconstructed_sha256": recon_sha,
            "diff_pixel_count": diff_count,
            "total_pixels": int(original.size / original.shape[-1]),
            "mse": mse,
            "psnr_db": psnr,
            "fidelity_percentage": 100.0 if exact_match else round((1.0 - (diff_count / original.size)) * 100.0, 4)
        }
