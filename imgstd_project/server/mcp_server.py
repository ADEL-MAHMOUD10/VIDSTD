"""
IMGSTD Reference Model Context Protocol (MCP) Server.
Exposes standard resources and tools for Zero-Vision AI image comprehension and bit-exact decoding.
"""

import os
import sys

# Ensure root workspace directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from typing import Optional, Dict, Any
import numpy as np

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from imgstd_project.core.inspector import inspect_imgstd
from imgstd_project.core.decoder import decode_imgstd
from imgstd_project.core.encoder import encode_image
from imgstd_project.core.video_codec import (
    encode_video_to_vidstd,
    decode_vidstd_to_video,
    extract_vidstd_frame,
    inspect_vidstd,
    get_vidstd_overview,
    search_vidstd_timeline,
    inspect_vidstd_timeframe,
)

# Initialize the MCP Server
mcp = MCPServer(
    name="IMGSTD-Reference-Server"
)

SPEC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spec"))
VOCAB_PATH = os.path.join(SPEC_DIR, "vocabulary.json")
SPEC_PATH = os.path.join(SPEC_DIR, "protocol_spec.md")


@mcp.resource("imgstd://spec/v01")
def get_protocol_specification() -> str:
    """
    Exposes the master vocabulary mapping and protocol specification for IMGSTD V01.
    Allows any connected AI model to look up standard 4-digit class IDs and layout rules.
    """
    vocab_data = {}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

    spec_doc = ""
    if os.path.exists(SPEC_PATH):
        with open(SPEC_PATH, "r", encoding="utf-8") as f:
            spec_doc = f.read()

    combined = {
        "version": "IMGSTD::V01",
        "description": "IMGSTD Master Vocabulary & Protocol Specification",
        "vocabulary": vocab_data,
        "protocol_documentation": spec_doc
    }
    return json.dumps(combined, indent=2, ensure_ascii=False)


@mcp.tool()
def inspect_imgstd_code(imgstd_code: str) -> Dict[str, Any]:
    """
    Ultra-fast O(1) Zero-Vision AI inspection tool.
    Parses ONLY the header [HDR] and semantic [SEM] layers in <2ms without Computer Vision,
    returning dimensions, SHA-256 seal, and all detected objects with their human-readable labels,
    bounding boxes, dominant RGB colors, and canvas coverage percentages.
    
    Args:
        imgstd_code: The raw IMGSTD string or a local file path to an .imgstd file.
    """
    return inspect_imgstd(imgstd_code)


@mcp.tool()
def decode_imgstd_to_image(imgstd_code: str, output_image_path: str, verify_sha: bool = True) -> Dict[str, Any]:
    """
    Restores an exact original image from an IMGSTD stream.
    Rebuilds the Base Canvas from [SEM], decompresses [RES], adds (B + R) mod 256,
    saves the image (PNG/BMP) to output_image_path, and verifies SHA-256 equality.

    Args:
        imgstd_code: The raw IMGSTD string or a local file path to an .imgstd file.
        output_image_path: Target path to write the restored image (e.g. 'restored.png').
        verify_sha: Set to False if decoding intentionally recolored/edited streams.
    """
    _, metrics = decode_imgstd(imgstd_code, output_image_path=output_image_path, verify_sha=verify_sha)
    return metrics


@mcp.tool()
def encode_image_to_imgstd(
    input_image_path: str,
    output_code_path: Optional[str] = None,
    scene_model: Optional[str] = None,
    enable_smolvlm2: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Encodes any digital image into a standardized Semantic-Lossless IMGSTD stream in milliseconds.
    Extracts semantic regions into [SEM] and compresses exact mathematical residual into [RES].

    Args:
        input_image_path: Path to the source image (PNG, JPG, BMP, etc.).
        output_code_path: Optional path to save the resulting .imgstd text stream.
        scene_model: Optional MobileCLIP2 model override (e.g. "mobileclip2-s0", "mobileclip2-s2", "mobileclip2-l-14").
        enable_smolvlm2: Optional override to run SmolVLM2 generative VLM captioning (true/false).
    """
    if not output_code_path:
        base = os.path.splitext(os.path.basename(input_image_path))[0]
        output_code_path = f"{base}.imgstd"

    code, metrics = encode_image(
        input_image_path,
        output_path=output_code_path,
        scene_model=scene_model,
        enable_smolvlm2=enable_smolvlm2,
    )
    return {
        "metrics": metrics,
        "output_path": output_code_path,
        "imgstd_code": code if len(code) < 32768 else output_code_path
    }


@mcp.tool()
def list_available_models() -> Dict[str, Any]:
    """
    Lists all available MobileCLIP2 variants and the optional SmolVLM2 generative VLM,
    including each model's hardware tier (LIGHTWEIGHT, BALANCED, HEAVY, GENERATIVE VLM),
    RAM/speed footprint, local download status, active state, and practical tips.
    """
    from imgstd_project.core.config import get_model_catalog
    return get_model_catalog()


@mcp.tool()
def get_settings() -> Dict[str, Any]:
    """
    Returns the current persistent IMGSTD/VIDSTD settings (scene_model, enable_smolvlm2,
    sample_fps, yolo_conf) along with tailored hardware recommendations and tips.
    """
    from imgstd_project.core.config import load_settings, get_hardware_tips, get_settings_path, detect_hardware_profile
    cfg = load_settings()
    return {
        "settings_path": get_settings_path(),
        "settings": cfg,
        "hardware": detect_hardware_profile(),
        "tips": get_hardware_tips(cfg),
    }


@mcp.tool()
def update_settings(
    scene_model: Optional[str] = None,
    enable_smolvlm2: Optional[bool] = None,
    sample_fps: Optional[float] = None,
    yolo_conf: Optional[float] = None,
    reset: bool = False,
    download_on_select: bool = True
) -> Dict[str, Any]:
    """
    Updates persistent model and performance settings for CLI and MCP runs.
    Automatically downloads newly selected model weights when download_on_select=True.

    Args:
        scene_model: MobileCLIP2 variant ID ("mobileclip2-s0", "mobileclip2-s2", "mobileclip2-b", "mobileclip2-s3", "mobileclip2-s4", "mobileclip2-l-14").
        enable_smolvlm2: Enable (true) or disable (false) SmolVLM2 generative VLM by default.
        sample_fps: Default sampling FPS for semantic_only video references (> 0).
        yolo_conf: Default YOLO detection confidence threshold (0.05 to 0.95).
        reset: Set true to restore lightweight CPU defaults.
        download_on_select: If True (default), downloads newly selected models so they are ready locally.
    """
    from imgstd_project.core.config import (
        save_settings,
        reset_settings,
        get_hardware_tips,
        get_settings_path,
        download_model_weights,
    )
    if reset:
        cfg = reset_settings()
        return {
            "updated": True,
            "settings_path": get_settings_path(),
            "settings": cfg,
            "downloaded_models": [],
            "tips": get_hardware_tips(cfg),
        }

    cfg = save_settings({
        "scene_model": scene_model,
        "enable_smolvlm2": enable_smolvlm2,
        "sample_fps": sample_fps,
        "yolo_conf": yolo_conf,
    })
    downloaded_models = []
    if download_on_select:
        if scene_model is not None:
            downloaded_models.append(download_model_weights(cfg["scene_model"], quiet=True))
        if enable_smolvlm2 is True:
            downloaded_models.append(download_model_weights("smolvlm2", quiet=True))
    return {
        "updated": True,
        "settings_path": get_settings_path(),
        "settings": cfg,
        "downloaded_models": downloaded_models,
        "tips": get_hardware_tips(cfg),
    }


def _load_stream_or_path(imgstd_code: str) -> str:
    """Safely loads a stream string or reads a UTF-8 file path without overflowing Windows path limits."""
    if not imgstd_code.startswith("IMGSTD::") and len(imgstd_code) < 4096 and os.path.exists(imgstd_code):
        with open(imgstd_code, "r", encoding="utf-8") as f:
            return f.read()
    return imgstd_code


def _recompute_imgstd_headers_and_crc(code: str) -> str:
    """Recomputes UTF-8 byte offsets in [HDR] and CRC32/TOTAL_BYTES in [END] after mutating [SEM]."""
    import re
    import zlib

    end_idx = code.find("\n[END]")
    partial = code[:end_idx] if end_idx != -1 else code.rstrip("\n")

    sem_start = partial.find("[SEM]")
    sem_end = partial.find("[/SEM]")
    if sem_start == -1 or sem_end == -1:
        return code
    sem_block = partial[sem_start:sem_end + 6]
    sem_bytes = sem_block.encode("utf-8")
    sem_len = len(sem_bytes)

    hdr_end = partial.find("[/HDR]")
    if hdr_end == -1:
        return code
    hdr_end_full = hdr_end + 6
    hdr_part = partial[:hdr_end_full]

    sem_off = len((hdr_part + "\n").encode("utf-8"))
    rest_after_sem = partial[sem_end + 6:]

    # Locate [TXT] and [RES] offsets inside rest_after_sem
    txt_start = partial.find("[TXT]")
    txt_end = partial.find("[/TXT]")
    if txt_start != -1 and txt_end != -1:
        txt_block = partial[txt_start:txt_end + 6]
        txt_len = len(txt_block.encode("utf-8"))
        txt_off = sem_off + sem_len + 1
        res_off = txt_off + txt_len + 1
    else:
        txt_off = 0
        txt_len = 0
        res_off = sem_off + sem_len + 1

    res_start = partial.find("[RES]")
    res_end = partial.find("[/RES]")
    res_len = len(partial[res_start:res_end + 6].encode("utf-8")) if (res_start != -1 and res_end != -1) else 0

    lm_rel = sem_bytes.find(b"LABEL_MAP_Z85:")
    if lm_rel != -1:
        lm_line_end = sem_bytes.find(b"\n", lm_rel)
        lm_len = (lm_line_end - lm_rel) if lm_line_end != -1 else (len(sem_bytes) - lm_rel)
        lm_off = sem_off + lm_rel
    else:
        lm_off = 0
        lm_len = 0

    replacements = {
        "SEM_OFFSET": f"{sem_off:010d}",
        "SEM_LEN": f"{sem_len:010d}",
        "TXT_OFFSET": f"{txt_off:010d}",
        "TXT_LEN": f"{txt_len:010d}",
        "RES_OFFSET": f"{res_off:010d}",
        "RES_LEN": f"{res_len:010d}",
        "LABELMAP_OFFSET": f"{lm_off:010d}",
        "LABELMAP_LEN": f"{lm_len:010d}",
    }
    new_hdr = hdr_part
    for key, val in replacements.items():
        new_hdr = re.sub(rf"({key}:)\d+", rf"\g<1>{val}", new_hdr)

    new_partial = new_hdr + partial[hdr_end_full:]
    partial_bytes = new_partial.encode("utf-8")
    crc32_val = f"{zlib.crc32(partial_bytes) & 0xFFFFFFFF:08x}"
    return f"{new_partial}\n[END]\nCRC32:{crc32_val}\nTOTAL_BYTES:{len(partial_bytes):010d}\n[/END]\n"


@mcp.tool()
def recolor_imgstd_region(imgstd_code: str, region_id: str, new_rgb: list) -> dict:
    """
    Surgical non-destructive region recoloring.
    Modifies ONLY the RGB base color of a specific region in [SEM] while preserving
    the exact [RES] lighting/texture layer, and updates [HDR] byte offsets + [END] CRC32.

    Args:
        imgstd_code: The IMGSTD code string or file path.
        region_id: The region ID to recolor (e.g. "001", "002").
        new_rgb: New [R, G, B] color values (0-255 each).
    """
    import re
    code = _load_stream_or_path(imgstd_code)

    r_val, g_val, b_val = int(new_rgb[0]), int(new_rgb[1]), int(new_rgb[2])

    # Match line starting with REG:{region_id}| and replace its RGB field
    pattern = rf"(REG:{re.escape(region_id)}\|(?:[^|]*\|)*?RGB:)\d+,\d+,\d+"
    new_code = re.sub(pattern, rf"\g<1>{r_val},{g_val},{b_val}", code)

    if new_code == code:
        return {"success": False, "error": f"Region {region_id} not found in [SEM] block"}

    new_code = _recompute_imgstd_headers_and_crc(new_code)

    return {
        "success": True,
        "region_id": region_id,
        "new_rgb": [r_val, g_val, b_val],
        "modified_code": new_code
    }


@mcp.tool()
def query_region_at_point(imgstd_code: str, x: int, y: int) -> dict:
    """
    Spatial Query API: Returns what semantic object occupies pixel coordinate (X, Y).
    Reads the LABEL_MAP from [SEM] for V02 streams, or bounding-box intersection for V01.

    Args:
        imgstd_code: The IMGSTD code string or file path.
        x: Pixel X coordinate (column).
        y: Pixel Y coordinate (row).
    """
    import base64
    from imgstd_project.core.inspector import inspect_imgstd

    code = _load_stream_or_path(imgstd_code)

    inspection = inspect_imgstd(code)

    # For V02, V03 & V04: check LABEL_MAP if present
    if any(ver in code for ver in ("IMGSTD::V04", "IMGSTD::V03", "IMGSTD::V02")):
        label_map_line = None
        for line in code.split('\n'):
            if line.startswith('LABEL_MAP_Z85:'):
                label_map_line = line[len('LABEL_MAP_Z85:'):]
                break

        if label_map_line:
            import zstandard as zstd
            H = inspection['dimensions']['height']
            W = inspection['dimensions']['width']
            compressed = base64.b85decode(label_map_line.encode('ascii'))
            label_raw = zstd.ZstdDecompressor().decompress(compressed)
            label_map = np.frombuffer(label_raw, dtype=np.uint8).reshape(H, W)

            if 0 <= y < H and 0 <= x < W:
                label_id = int(label_map[y, x])
                if label_id > 0 and label_id - 1 < len(inspection['regions']):
                    reg = inspection['regions'][label_id - 1]
                    res_dict = {
                        "x": x, "y": y,
                        "label_id": label_id,
                        "region_id": reg['region_id'],
                        "kind": reg.get('kind', 'OBJ'),
                        "confidence": reg.get('confidence', 1.00),
                        "role": reg.get('role', reg.get('prominence', 'SECONDARY')),
                        "prominence": reg.get('prominence', 'SECONDARY'),
                        "class_id": reg['class_id'],
                        "class_name": reg['class_name'],
                        "dominant_color_rgb": reg['dominant_color_rgb'],
                        "coverage_percentage": reg['coverage_percentage']
                    }
                    if reg.get('is_reflection') or reg.get('likely_reflection_or_inset'):
                        res_dict["is_reflection"] = True
                        res_dict["likely_reflection_or_inset"] = True
                        res_dict["parent_structure"] = reg.get("parent_structure")
                        res_dict["contained_in"] = reg.get("contained_in")
                    return res_dict

    # Fallback: bounding box intersection for V01
    matches = []
    for reg in inspection['regions']:
        bb = reg['bounding_box']
        if bb['x'] <= x < bb['x'] + bb['width'] and bb['y'] <= y < bb['y'] + bb['height']:
            matches.append(reg)

    if not matches:
        return {"x": x, "y": y, "kind": "BG", "confidence": 1.00, "class_name": "Unclassified Background / Surface", "class_id": "0900"}

    # Return smallest bounding box (most specific region)
    best = min(matches, key=lambda r: r['bounding_box']['width'] * r['bounding_box']['height'])
    return {
        "x": x, "y": y,
        "region_id": best['region_id'],
        "kind": best.get('kind', 'OBJ'),
        "confidence": best.get('confidence', 1.00),
        "class_id": best['class_id'],
        "class_name": best['class_name'],
        "dominant_color_rgb": best['dominant_color_rgb'],
    }


# =====================================================================
# VIDSTD Video Standard MCP Tools (V01)
# =====================================================================

@mcp.tool()
def encode_video_to_vidstd_tool(
    video_path: str,
    output_vidstd_path: Optional[str] = None,
    gop_size: int = 24,
    semantic_only: bool = False,
    sample_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    scene_model: Optional[str] = None,
    enable_smolvlm2: Optional[bool] = None
) -> Dict[str, Any]:
    """
    High-Throughput Lossless Video Encoder & Lightweight AI Reference Builder.
    Compresses any video (.mp4, .avi, etc.) into a VIDSTD::V01 container, or generates a tiny
    SEMANTIC_REF index when semantic_only=True.

    Args:
        video_path: Path to source video file.
        output_vidstd_path: Destination .vidstd path (default: <name>.vidstd).
        gop_size: Group of Pictures size (default: 24 frames = 1 second @ 24fps).
        semantic_only: If True, builds an ultra-fast lightweight AI reference (<1 MB, no pixel payload).
        sample_fps: Optional sampling FPS when semantic_only=True (default: from settings / 2.0).
        max_frames: Optional cap on processed frames.
        scene_model: Optional MobileCLIP2 model override (e.g. "mobileclip2-s0", "mobileclip2-s2", "mobileclip2-l-14").
        enable_smolvlm2: Optional override to run SmolVLM2 generative VLM (true/false).
    """
    if not output_vidstd_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_vidstd_path = f"{base}.vidstd"

    _, metrics = encode_video_to_vidstd(
        video_path,
        output_path=output_vidstd_path,
        gop_size=gop_size,
        max_frames=max_frames,
        semantic_only=semantic_only,
        sample_fps=sample_fps,
        scene_model=scene_model,
        enable_smolvlm2=enable_smolvlm2,
    )
    metrics["output_path"] = output_vidstd_path
    return metrics


@mcp.tool()
def inspect_vidstd_timeline(vidstd_path: str) -> Dict[str, Any]:
    """
    Sub-millisecond Zero-Vision Full-Video AI Comprehension (< 1 ms).
    Reads ONLY [VID_HDR] and [TIMELINE_INDEX] without decoding pixels,
    returning structured scenes, object trajectories, text events, and duration.

    Args:
        vidstd_path: Path to .vidstd file.
    """
    return inspect_vidstd(vidstd_path)


@mcp.tool()
def query_vidstd_at_timestamp(vidstd_path: str, timestamp_sec: float) -> Dict[str, Any]:
    """
    Spatial & Temporal Query API: Returns all active objects, bounding boxes,
    visible texts, and current scene at second T in the video.

    Args:
        vidstd_path: Path to .vidstd file.
        timestamp_sec: Video timestamp in seconds (e.g. 5.5).
    """
    inspection = inspect_vidstd(vidstd_path)
    fps = inspection["video_metadata"]["fps"]
    target_frame = int(round(timestamp_sec * fps))

    # Find active scene
    active_scene = None
    for sc in inspection.get("scenes", []):
        frames_part = sc.get("frames", "0-0")
        if "-" in frames_part:
            sf, ef = [int(v) for v in frames_part.split("-")]
            if sf <= target_frame <= ef:
                active_scene = sc
                break

    # Find active objects
    active_objects = []
    for trk in inspection.get("tracked_objects", []):
        ff_str = trk.get("first_frame", "0(0.00s)")
        lf_str = trk.get("last_frame", "0(0.00s)")
        ff = int(ff_str.split("(")[0]) if "(" in ff_str else int(ff_str)
        lf = int(lf_str.split("(")[0]) if "(" in lf_str else int(lf_str)

        if ff <= target_frame <= lf:
            active_objects.append({
                "track_id": trk["track_id"],
                "class": trk["class"],
                "role": trk.get("role", "SECONDARY"),
                "path": trk.get("path"),
                "dominant_rgb": trk.get("dominant_rgb"),
                "is_reflection": bool(trk.get("inside_of")),
                "inside_of": trk.get("inside_of")
            })

    # Find active texts
    active_texts = []
    for tt in inspection.get("text_events", []):
        ff_str = tt.get("first_frame", "0(0.00s)")
        lf_str = tt.get("last_frame", "0(0.00s)")
        ff = int(ff_str.split("(")[0]) if "(" in ff_str else int(ff_str)
        lf = int(lf_str.split("(")[0]) if "(" in lf_str else int(lf_str)

        if ff <= target_frame <= lf:
            active_texts.append({
                "text": tt["text"],
                "box": tt.get("box")
            })

    return {
        "timestamp_sec": timestamp_sec,
        "frame_index": target_frame,
        "active_scene": active_scene,
        "active_objects_count": len(active_objects),
        "active_objects": active_objects,
        "active_texts": active_texts
    }


@mcp.tool()
def extract_lossless_video_frame(vidstd_path: str, frame_index: int, output_image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    O(1) Random-Access Bit-Exact Frame Extraction (< 35 ms).
    Seeks directly to the target GOP via GOP_OFFSET_TABLE, decodes the frame,
    verifies the SHA-256 seal, and optionally saves the frame to disk.

    Args:
        vidstd_path: Path to .vidstd file.
        frame_index: 0-indexed integer frame number to extract.
        output_image_path: Optional path to save extracted frame (e.g. 'frame_45.png').
    """
    _, metrics = extract_vidstd_frame(vidstd_path, frame_index, output_image_path=output_image_path)
    if output_image_path:
        metrics["output_path"] = output_image_path
    return metrics


@mcp.tool()
def get_vidstd_overview_tool(vidstd_path: str) -> Dict[str, Any]:
    """
    Hierarchical Level 1 Video Query: Instant (<5ms), low-token (<500 tokens) overview
    of video scenes, primary subjects, detected object classes, and signage texts.
    Ideal for Zero-Vision LLM orientation before diving into specific scenes.

    Args:
        vidstd_path: Path to .vidstd file.
    """
    return get_vidstd_overview(vidstd_path)


@mcp.tool()
def search_vidstd_timeline_tool(
    vidstd_path: str,
    query: Optional[str] = None,
    class_filter: Optional[str] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Hierarchical Level 2 Video Query: Targeted search across scenes, entities, and OCR text
    with optional keyword, class name, and time-range filtering.

    Args:
        vidstd_path: Path to .vidstd file.
        query: Substring to match in object class, motion, or text events.
        class_filter: Filter for specific class (e.g. '0201', 'Planet Earth', 'Car').
        start_sec: Optional start time in seconds.
        end_sec: Optional end time in seconds.
    """
    return search_vidstd_timeline(
        vidstd_path,
        query=query,
        class_filter=class_filter,
        start_sec=start_sec,
        end_sec=end_sec
    )


@mcp.tool()
def inspect_vidstd_timeframe_tool(
    vidstd_path: str,
    start_sec: float,
    end_sec: float
) -> Dict[str, Any]:
    """
    Hierarchical Level 3 Video Query: Deep inspection of a specific temporal interval [start_sec, end_sec].
    Returns active scenes, active objects, on-screen text, and the nearest I-Frame keyframe index
    for instant O(1) single-frame extraction.

    Args:
        vidstd_path: Path to .vidstd file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
    """
    return inspect_vidstd_timeframe(vidstd_path, start_sec=start_sec, end_sec=end_sec)


def main():
    """CLI entry point for the IMGSTD/VIDSTD MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
