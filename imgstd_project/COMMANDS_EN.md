# IMGSTD & VIDSTD (v1.0) — Global CLI & API Reference Guide

Production-ready command-line codec & Non-Vision AI reference toolkit (`vidstd` / `imgstd`), usable globally from any directory just like `ffmpeg`.

---

## 1. Setup & Global Installation

Install dependencies and register the `vidstd`, `imgstd`, and `imgstd-mcp` commands system-wide:
```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```
Verify from any directory:
```bash
vidstd --version
vidstd --help
```

---

## 2. Video Commands (`VIDSTD`)

### A) Build Fast Lightweight AI Reference (`< 1 MB`, ~2 min for 5-min 1080p on CPU)
When you already have the original video and want a structured semantic reference for any Non-Vision LLM:
```bash
vidstd vid-ref "video.mp4" -o "video_ref.vidstd"
```

### B) Sub-Millisecond Zero-Vision Inspection (`--llm` / `--json`)
```bash
vidstd vid-inspect "video_ref.vidstd" --llm
vidstd vid-inspect "video_ref.vidstd" --json
```

### C) Full Bit-Exact Lossless Video Compression & Decoding (`O(1)` Streaming RAM)
```bash
vidstd vid-encode "clip.mp4" -o "clip.vidstd"
vidstd vid-decode "clip.vidstd" -o "clip_restored.mp4"
vidstd vid-frame "clip.vidstd" --frame 45 -o "frame45.png"
```

---

## 3. Image Commands (`IMGSTD`)

```bash
vidstd encode "image.png" -o "image.imgstd"
vidstd inspect "image.imgstd" --llm
vidstd decode "image.imgstd" -o "restored.png"
vidstd roundtrip "image.png"
```

---

# Model Selection & First-Time Setup (`MobileCLIP2` & `SmolVLM2`)

Inspect all available `MobileCLIP2` tiers and the optional `SmolVLM2` generative VLM, see hardware tips and specs next to each model, or pre-download / switch defaults (selecting a model in `config set` downloads its weights automatically unless `--no-download` is passed):

```bash
# Pre-download core vision models (YOLO11n-seg, YOLO-World, RapidOCR, active MobileCLIP2):
vidstd setup
vidstd setup --model mobileclip2-s2 --smolvlm2

# List all models with tier tags (LIGHTWEIGHT / BALANCED / HEAVY / GENERATIVE VLM) & tips
vidstd models
vidstd models --json

# Inspect or update saved settings (auto-downloads selected models)
vidstd config show
vidstd config set --scene-model mobileclip2-s2 --smolvlm2 off
vidstd config set --smolvlm2 on --sample-fps 2.0 --yolo-conf 0.40
vidstd config reset

# Override model per run without changing saved settings:
vidstd vid-ref "video.mp4" --model mobileclip2-b -o "ref.vidstd"
vidstd encode "image.png" --model mobileclip2-s2 --smolvlm2 -o "image.imgstd"
```

---

## 4. Quick Verification & Interactive Demo

### A) End-to-End Image Roundtrip Verification
Runs real encoding, semantic inspection, bit-exact decoding, and SHA-256 / `MSE = 0.0` verification on any image:
```bash
vidstd roundtrip "image.png"
```

### B) Interactive Quickstart Demo
Runs an end-to-end demonstration on a sample scene with printed layer snippets:
```bash
python imgstd_project/demo.py
```

---

## 4. Model Context Protocol (MCP) Server

Start the FastMCP server to integrate IMGSTD with any AI assistant or agent:
```bash
python imgstd_project/server/mcp_server.py
```

### Exposed MCP Tools & Resources:
* `imgstd://spec/v01` *(Resource)*: Master standard vocabulary (classes 0000–0999) & protocol specification.
* `inspect_imgstd_code` *(Tool)*: Inspects scene semantics, objects, and layout in $<1\text{ms}$.
* `decode_imgstd_to_image` *(Tool)*: Restores 100% bit-exact original image and verifies SHA-256 checksum.
* `encode_image_to_imgstd` *(Tool)*: Encodes any digital image into IMGSTD V04 (accepts `scene_model` & `enable_smolvlm2` overrides).
* `encode_video_to_vidstd_tool` *(Tool)*: Encodes video or builds lightweight AI references (`semantic_only=True`, `scene_model`, `enable_smolvlm2`).
* `list_available_models` *(Tool)*: Returns all MobileCLIP2 variants + SmolVLM2 with tiers, RAM/CPU specs, local availability, and hardware tips.
* `get_settings` / `update_settings` *(Tools)*: Reads or updates persistent model & performance preferences directly from an AI agent.
* `recolor_imgstd_region` *(Tool)*: Non-destructive region recoloring updating byte offsets & CRC32 cleanly.
* `query_region_at_point` *(Tool)*: Spatial query returning the semantic object at coordinates `(X, Y)`.

---

## 5. Python API Code Snippets

### Standard Roundtrip Pipeline:
```python
from imgstd_project.core.encoder import encode_image
from imgstd_project.core.decoder import decode_imgstd
from imgstd_project.core.inspector import inspect_imgstd
from imgstd_project.core.verifier import IntegrityVerifier

# 1. Encode image
code, enc_metrics = encode_image("input.png", output_path="compressed.imgstd")
print(f"Encoded in {enc_metrics['total_encode_time_ms']} ms")

# 2. Inspect semantics in <1ms
summary = inspect_imgstd(code)
print(f"Detected {summary['region_count']} regions in {summary['inspection_time_ms']} ms")

# 3. Decode back to 100% bit-exact image
restored_img, dec_metrics = decode_imgstd("compressed.imgstd", output_image_path="restored.png")
print(f"Decoded in {dec_metrics['total_decode_time_ms']} ms")

# 4. Mathematical verification
audit = IntegrityVerifier.verify_bit_exact(original_img, restored_img)
assert audit["verified"] is True
assert audit["mse"] == 0.0
```

### Spatial Point Query API:
```python
from imgstd_project.server.mcp_server import query_region_at_point

# Query what object is located at pixel coordinates (X=250, Y=180)
result = query_region_at_point("compressed.imgstd", x=250, y=180)
print(f"Object: {result['class_name']} (Class ID: {result['class_id']})")
```

### Surgical Non-Destructive Region Recoloring:
```python
from imgstd_project.server.mcp_server import recolor_imgstd_region

# Change region "001" base color to pure red [255, 0, 0] without altering lighting/texture
result = recolor_imgstd_region("compressed.imgstd", region_id="001", new_rgb=[255, 0, 0])
with open("recolored.imgstd", "w") as f:
    f.write(result["modified_code"])
```

### Video Temporal Delta Encoding (VIDSTD Prototype):
```python
from imgstd_project.core.video_delta import encode_delta_frame, decode_delta_frame

# Encode temporal changes between consecutive video frames
delta_code, delta_metrics = encode_delta_frame(previous_frame, current_frame)

# Decode current frame using only previous reference frame + delta
restored_frame, _ = decode_delta_frame(delta_code, previous_frame)
```
