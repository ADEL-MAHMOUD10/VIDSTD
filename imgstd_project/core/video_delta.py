"""
IMGSTD VIDSTD Video Delta Module.
Encodes temporal delta frames between consecutive images for efficient video encoding.
Bit-exact: decode_delta_frame(encode_delta_frame(prev, curr)[0], prev)[0] == curr.
"""

import base64
import time
from typing import Tuple, Dict, Any

import numpy as np
import zstandard as zstd


# 16x16 macro-block size for block-based delta detection
BLOCK_SIZE = 16


def encode_delta_frame(
    prev: np.ndarray,
    curr: np.ndarray,
    compression_level: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """
    Encode the temporal delta between *prev* and *curr* frames.

    Strategy
    --------
    1. Divide both frames into BLOCK_SIZE×BLOCK_SIZE macro-blocks.
    2. For each block, compute whether it changed (any pixel differs).
    3. Build a bit-mask of changed blocks (compact, stored as bytes).
    4. Collect the raw pixels of ONLY changed blocks into a payload buffer.
    5. Zstd-compress → base85-encode the whole thing for ASCII transport.

    Returns a compact stream string and a metrics dict with 'changed_blocks'.
    """
    t_start = time.perf_counter()

    if prev.shape != curr.shape:
        raise ValueError(
            f"Frame shape mismatch: prev={prev.shape}, curr={curr.shape}"
        )

    h, w = prev.shape[:2]
    c = prev.shape[2] if prev.ndim == 3 else 1

    # Number of blocks in each dimension
    blocks_y = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    blocks_x = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = blocks_y * blocks_x

    changed_flags = np.zeros(total_blocks, dtype=np.uint8)
    changed_data_parts = []

    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0, y1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, h)
            x0, x1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, w)
            blk_prev = prev[y0:y1, x0:x1]
            blk_curr = curr[y0:y1, x0:x1]
            idx = by * blocks_x + bx
            if not np.array_equal(blk_prev, blk_curr):
                changed_flags[idx] = 1
                # Store: block position metadata + raw curr pixels
                blk_h = y1 - y0
                blk_w = x1 - x0
                # Header: 4 uint16 LE values: by, bx, blk_h, blk_w
                header = np.array([by, bx, blk_h, blk_w], dtype=np.uint16)
                changed_data_parts.append(header.tobytes())
                changed_data_parts.append(blk_curr.tobytes())

    changed_blocks = int(np.sum(changed_flags))

    # Build payload: [changed_flags bytes] + [all changed block data]
    payload = changed_flags.tobytes() + b"".join(changed_data_parts)

    # Compress + base85-encode
    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed = cctx.compress(payload)
    b85_str = base64.b85encode(compressed).decode("ascii")

    # Stream: "VIDSTD_DELTA\nH:h\nW:w\nC:c\nBY:blocks_y\nBX:blocks_x\n<data>"
    stream = (
        f"VIDSTD_DELTA\n"
        f"H:{h}\nW:{w}\nC:{c}\n"
        f"BY:{blocks_y}\nBX:{blocks_x}\n"
        f"CHANGED:{changed_blocks}\n"
        f"DATA:{b85_str}"
    )

    t_ms = (time.perf_counter() - t_start) * 1000.0
    metrics: Dict[str, Any] = {
        "changed_blocks": changed_blocks,
        "total_blocks": total_blocks,
        "changed_ratio": round(changed_blocks / max(1, total_blocks), 4),
        "encode_time_ms": round(t_ms, 3),
    }
    return stream, metrics


def decode_delta_frame(
    delta_stream: str,
    prev: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reconstruct *curr* from *prev* and *delta_stream*.
    Guaranteed to produce bit-exact *curr* for any (prev, curr) pair.
    """
    t_start = time.perf_counter()

    lines = delta_stream.splitlines()
    if not lines or lines[0] != "VIDSTD_DELTA":
        raise ValueError("Invalid VIDSTD_DELTA stream: bad magic.")

    meta: Dict[str, str] = {}
    data_b85 = None
    for line in lines[1:]:
        if line.startswith("DATA:"):
            data_b85 = line[5:]
        elif ":" in line:
            k, v = line.split(":", 1)
            meta[k] = v

    if data_b85 is None:
        raise ValueError("VIDSTD_DELTA stream missing DATA section.")

    h, w, c = int(meta["H"]), int(meta["W"]), int(meta["C"])
    blocks_y, blocks_x = int(meta["BY"]), int(meta["BX"])
    total_blocks = blocks_y * blocks_x

    # Decompress payload
    dctx = zstd.ZstdDecompressor()
    payload = dctx.decompress(base64.b85decode(data_b85.encode("ascii")))

    # Split into changed_flags and block data
    flags_bytes = payload[:total_blocks]
    changed_flags = np.frombuffer(flags_bytes, dtype=np.uint8)

    # Start from copy of prev
    curr = prev.copy()

    # Reconstruct changed blocks
    offset = total_blocks  # byte offset into payload
    header_size = 4 * 2  # 4 × uint16 = 8 bytes

    for idx in range(total_blocks):
        if changed_flags[idx] == 0:
            continue
        # Read header
        header = np.frombuffer(payload[offset: offset + header_size], dtype=np.uint16)
        offset += header_size
        by_i, bx_i, blk_h, blk_w = int(header[0]), int(header[1]), int(header[2]), int(header[3])
        # Read pixel data
        blk_bytes = blk_h * blk_w * c
        blk_data = np.frombuffer(payload[offset: offset + blk_bytes], dtype=np.uint8)
        offset += blk_bytes
        blk = blk_data.reshape((blk_h, blk_w, c) if c > 1 else (blk_h, blk_w))
        y0, x0 = by_i * BLOCK_SIZE, bx_i * BLOCK_SIZE
        curr[y0: y0 + blk_h, x0: x0 + blk_w] = blk

    t_ms = (time.perf_counter() - t_start) * 1000.0
    metrics_out: Dict[str, Any] = {
        "decode_time_ms": round(t_ms, 3),
        "total_blocks": total_blocks,
    }
    return curr, metrics_out
