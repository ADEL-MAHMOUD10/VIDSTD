<div align="center">

![IMGSTD & VIDSTD — Let Any Non-Vision Model See Images & Video](assets/hero-banner.jpg)

# IMGSTD & VIDSTD (v1.0)

**Ultra-Fast Semantic-Lossless Media Codec & Non-Vision AI Reference CLI / MCP Toolkit**

</div>

`IMGSTD` & `VIDSTD` convert images and videos into compact, human/AI-readable **semantic representations** (`[SEM]`, `[TXT]`, `[ATTR]`, `[TIMELINE_INDEX]`, `[EGO]`, `[POSE]`, `[ACTION]`) readable by any text-only LLM in **< 1 ms**, while optionally keeping a bit-exact mathematical residual (`MSE = 0.0`, SHA-256 verified) or creating ultra-light **< 1 MB Semantic Reference** streams (`vid-ref`).

---

## Architecture Overview

![IMGSTD / VIDSTD Architecture — Giving Text-Only AI Ground-Truth Vision](assets/architecture-overview.jpg)

### System Pipeline & Protocol Diagram

![IMGSTD & VIDSTD Technical Pipeline Diagram](assets/system-architecture-diagram.png)

---

## Key Capabilities

- **Hybrid Instance Segmentation + Open-Vocabulary Architecture**:
  - `YOLO11n-seg` (`yolo11n-seg.pt`) for high-recall pixel segmentation masks and class-agnostic deduplication (`agnostic_nms=True`).
  - Targeted `YOLO-World` (`yolov8s-worldv2.pt`) non-overlapping supplement (`heavy machinery`, `spaceship`, `soldier`, `rodent`, `chart`, `diagram`) without COCO class dilution.
  - Perceptual HSV color naming (`burnt orange / rust-orange`, `golden tan / light brown`, `dark navy / denim blue`), clothing upper/lower analysis, and traffic light signal state detection.
- **Zero-Vision AI Inspection (< 1 ms)**:
  - Extract grounded Markdown (`--llm`) or JSON (`--json`) reports with zero image/video pixel decoding at read time.
- **Configurable Scene & Generative Models (`MobileCLIP2` & `SmolVLM2`)**:
  - Choose from 6 `MobileCLIP2` tiers (`s0`, `s2`, `b`, `s3`, `s4`, `l-14`) plus optional `SmolVLM2-500M-Video-Instruct`.
  - **Auto-download on configuration**: switching models via `vidstd config set` or MCP `update_settings` automatically fetches and caches weights locally.
- **Bit-Exact Lossless Reconstruction (`MSE = 0.0`)**:
  - Verify cryptographic parity (`SHA-256` & pixel equality) via `vidstd roundtrip` / `vidstd vid-roundtrip`.
- **Native MCP Server (`imgstd-mcp`)**:
  - Plug directly into Claude Desktop, Cursor, Windsurf, or Antigravity as an MCP server.

---

## Quick Start (Fresh Clone)

### 1. Clone & Install

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_DIR>
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

Verify global CLI registration from any folder:

```bash
vidstd --version
vidstd --help
```

### 2. Initialize Models (One-Time Setup or Auto-Download on Demand)

Model weights are excluded from Git (`weights/` is gitignored) and downloaded cleanly on demand into `./weights` (or `~/.imgstd/weights` when installed outside a repo):

```bash
# Optional one-command setup for base models (YOLO11n-seg + YOLO-World + RapidOCR + active MobileCLIP2):
vidstd setup

# Or also pre-download a specific MobileCLIP2 variant and/or SmolVLM2:
vidstd setup --model mobileclip2-s2 --smolvlm2
```

Even without `vidstd setup`, weights are **automatically downloaded** when you select a model in settings or run your first encode command.

---

## Choosing & Switching Models (Auto-Download on Configure)

```bash
# View all MobileCLIP2 tiers, SmolVLM2, local cache status, RAM/speed metrics & hardware tips:
vidstd models

# Switch active model (automatically downloads & builds prompt cache for that model):
vidstd config set --scene-model mobileclip2-s2

# Enable & automatically download SmolVLM2-500M-Video-Instruct (~1 GB):
vidstd config set --smolvlm2 on

# Save setting without downloading immediately:
vidstd config set --scene-model mobileclip2-b --no-download

# View current configuration or reset to lightweight CPU defaults:
vidstd config show
vidstd config reset
```

---

## Everyday CLI Usage

### Videos (`vid-ref`, `vid-inspect`, `vid-encode`, `vid-decode`)

```bash
# Build lightweight AI semantic reference (<1 MB, fast on CPU) from an existing video:
vidstd vid-ref "video.mp4" -o "video_ref.vidstd"

# Sub-millisecond context extraction for any text-only LLM:
vidstd vid-inspect "video_ref.vidstd" --llm
vidstd vid-inspect "video_ref.vidstd" --json

# Full bit-exact lossless compression & frame/video recovery:
vidstd vid-encode "clip.mp4" -o "clip.vidstd"
vidstd vid-decode "clip.vidstd" -o "clip_restored.mkv"
vidstd vid-frame "clip.vidstd" --frame 45 -o "frame45.png"
vidstd vid-roundtrip "clip.mp4"
```

### Images (`encode`, `inspect`, `decode`, `roundtrip`)

```bash
vidstd encode "photo.jpg" -o "photo.imgstd"
vidstd inspect "photo.imgstd" --llm
vidstd decode "photo.imgstd" -o "restored.png"
vidstd roundtrip "photo.jpg"
```

*(Note: `vidstd` and `imgstd` are identical aliases.)*

---

## MCP Server Integration

Start the MCP server via script entry point or Python:

```bash
imgstd-mcp
# or:
python -m imgstd_project.server.mcp_server
```

Example MCP client config (`claude_desktop_config.json` / `.mcp.json`):

```json
{
  "mcpServers": {
    "imgstd-vidstd": {
      "command": "imgstd-mcp"
    }
  }
}
```

Key MCP tools exposed:
- `list_available_models`, `get_settings`, `update_settings` (with auto-download when switching models)
- `encode_image_to_imgstd`, `inspect_imgstd_code`, `query_region_at_point`, `recolor_imgstd_region`, `decode_imgstd_to_image`
- `encode_video_to_vidstd_tool`, `inspect_vidstd_timeline`, `query_vidstd_at_timestamp`, `extract_lossless_video_frame`, `decode_vidstd_to_video_tool`

---

## Additional Documentation

- [CLI & API Reference (English)](imgstd_project/COMMANDS_EN.md)
- [دليل الأوامر (العربية)](imgstd_project/COMMANDS.md)
- [Protocol Specification](imgstd_project/spec/protocol_spec.md)
- [AI Agent System Prompt](imgstd_project/server/AI_SYSTEM_PROMPT.md)
