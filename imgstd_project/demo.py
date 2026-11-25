"""
IMGSTD Quickstart & Interactive Demonstration Script.
Demonstrates encoding, zero-vision inspection, bit-exact restoration, and verification.
"""

import os
import sys
import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imgstd_project.core.encoder import encode_image
from imgstd_project.core.inspector import inspect_imgstd
from imgstd_project.core.decoder import decode_imgstd
from imgstd_project.core.verifier import IntegrityVerifier


def run_demo():
    print("=" * 70)
    print("  IMGSTD (Standard Semantic-Lossless Codec) Demonstration")
    print("=" * 70)

    # 1. Create a sample 640x480 test image
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:160, :] = [135, 206, 235]       # Sky (Blue)
    img[160:340, :] = [34, 139, 34]      # Grass & Foliage (Green)
    img[340:, :] = [60, 60, 60]          # Asphalt Road (Dark Gray)
    img[240:360, 200:420] = [220, 20, 60] # Vehicle (Red)

    # Add realistic sensor noise and subtle lighting gradient
    for y in range(h):
        img[y, :, 0] = np.clip(img[y, :, 0].astype(int) + (y % 15) - 7, 0, 255)

    sample_img_path = "sample_input.png"
    cv2.imwrite(sample_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\n[1] Generated sample image: {sample_img_path} ({w}x{h}, 3 channels)")

    # 2. Encode Image -> IMGSTD Code
    output_code_path = "sample_image.imgstd"
    print("\n[2] Encoding to IMGSTD format...")
    code, enc_metrics = encode_image(sample_img_path, output_path=output_code_path)
    print(f"    - Encoding Latency: {enc_metrics['total_encode_time_ms']} ms")
    print(f"    - Original Raw Size: {enc_metrics['original_bytes']:,} bytes")
    print(f"    - Serialized Stream Size: {enc_metrics['encoded_bytes']:,} bytes")
    print(f"    - Compression Ratio: {enc_metrics['compression_ratio']}x")
    print(f"    - SHA-256 Hash: {enc_metrics['sha256']}")

    print("\n--- [IMGSTD Code Preview (Header & Semantic Layers)] ---")
    preview_lines = code.split("\n")[:18]
    print("\n".join(preview_lines))
    print("    ... [Residual Stream Layer [RES] Omitted] ...\n")

    # 3. Zero-Vision AI Inspection (Sub-Millisecond)
    print("[3] Running Zero-Vision AI Inspection (<2ms)...")
    inspection = inspect_imgstd(output_code_path)
    print(f"    - Inspection Latency: {inspection['inspection_time_ms']} ms")
    print(f"    - Detected Regions: {inspection['region_count']}")
    for r in inspection["regions"]:
        print(f"      * [{r['class_id']}] {r['class_name']} ({r['category']}):")
        print(f"        Bounding Box: [X={r['bounding_box']['x']}, Y={r['bounding_box']['y']}, W={r['bounding_box']['width']}, H={r['bounding_box']['height']}]")
        print(f"        Dominant RGB: {r['dominant_color_rgb']} | Canvas Coverage: {r['coverage_percentage']}%")

    # 4. Bit-Exact Decoding
    restored_img_path = "restored_output.png"
    print("\n[4] Reconstructing 100% Bit-Exact Image from IMGSTD...")
    restored_img, dec_metrics = decode_imgstd(output_code_path, output_image_path=restored_img_path)
    print(f"    - Decoding Latency: {dec_metrics['total_decode_time_ms']} ms")
    print(f"    - Restored File: {restored_img_path}")
    print(f"    - Reconstructed SHA-256: {dec_metrics['sha256']}")

    # 5. Cryptographic & Pixel Integrity Audit
    print("\n[5] Cryptographic & Mathematical Integrity Audit:")
    audit = IntegrityVerifier.verify_bit_exact(img, restored_img)
    print(f"    - Verified Bit-Exact: {audit['verified']}")
    print(f"    - SHA-256 Checksum Match: {audit['sha256_match']}")
    print(f"    - Mean Squared Error (MSE): {audit['mse']}")
    print(f"    - Peak Signal-to-Noise Ratio (PSNR): {audit['psnr_db']} dB")
    print(f"    - Differing Pixels: {audit['diff_pixel_count']} / {audit['total_pixels']}")
    print(f"    - Mathematical Fidelity: {audit['fidelity_percentage']}%")

    assert audit["verified"] is True
    print("\n" + "=" * 70)
    print("  ALL TESTS & CHECKS PASSED: 100.000% LOSSLESS RECONSTRUCTION CONFIRMED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
