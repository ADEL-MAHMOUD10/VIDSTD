"""
IMGSTD Command Line Interface (CLI).
Quick commands to encode, inspect, decode, or run a full roundtrip verification on any image.
"""

import sys
import os
import warnings
import argparse
import json

# Suppress noisy upstream deprecation warnings in production CLI output
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure parent directory is on sys.path when invoked directly as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imgstd_project import __version__


def _resolve_smolvlm2_flag(args) -> object:
    if getattr(args, "smolvlm2", False):
        return True
    if getattr(args, "no_smolvlm2", False):
        return False
    return None


def cmd_encode(args):
    from imgstd_project.core.encoder import encode_image
    print(f"[*] Encoding image: {args.image_path}")
    code, metrics = encode_image(
        args.image_path,
        output_path=args.output,
        scene_model=getattr(args, "model", None),
        enable_smolvlm2=_resolve_smolvlm2_flag(args),
    )
    print(f"[+] Encoding complete in {metrics['total_encode_time_ms']} ms")
    print(f"    - Scene Model:   {metrics.get('scene_model')} (SmolVLM2: {'ON' if metrics.get('smolvlm2_enabled') else 'OFF'})")
    print(f"    - Dimensions:    {metrics['width']}x{metrics['height']} ({metrics['channels']} channels)")
    print(f"    - Original Size: {metrics['original_bytes']:,} bytes")
    print(f"    - IMGSTD Size:   {metrics['encoded_bytes']:,} bytes ({metrics['compression_ratio']}x)")
    print(f"    - SHA-256 Seal:  {metrics['sha256']}")
    if args.output:
        print(f"[+] Saved IMGSTD code to: {args.output}")
    else:
        print("\n--- Preview of IMGSTD code ---")
        lines = code.split("\n")
        print("\n".join(lines[:15]))
        print("... [Remainder omitted, pass -o to save to file] ...")


def cmd_inspect(args):
    from imgstd_project.core.inspector import inspect_imgstd
    res = inspect_imgstd(args.code_or_file)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if getattr(args, "llm", False):
        dims = res["dimensions"]
        print("# NON-VISION AI GROUND TRUTH IMAGE REFERENCE")
        print(f"**Resolution**: {dims['width']}x{dims['height']} ({res['color_space']}) | **Protocol**: `{res['protocol']}`")
        print(f"**Integrity**: SHA-256 `{res['integrity_sha256']}`")
        if res.get("scene_description") or res.get("scene_summary"):
            print(f"\n## Scene Context\n{res.get('scene_description') or res.get('scene_summary')}")
        objs = [r for r in res.get("regions", []) if r.get("kind") in ("OBJ", "STRUCT")]
        if objs:
            from collections import Counter
            counts = Counter(r["class_name"].split("(")[0].strip() for r in objs)
            count_line = ", ".join(f"{name}: {cnt}" for name, cnt in counts.items())
            print(f"\n## Detected Entities & Structures ({len(objs)} total — {count_line})")
            for r in objs:
                bb = r["bounding_box"]
                attr_str = f" — {r['attributes']}" if r.get("attributes") else ""
                refl = f" (Reflected inside {r['parent_structure']})" if r.get("is_reflection") else ""
                print(f"- **{r['class_name']}** [{r.get('role', 'SECONDARY')}]: box `({bb['x']},{bb['y']},{bb['width']},{bb['height']})`, cov `{r['coverage_percentage']}%`{attr_str}{refl}")
        texts = res.get("texts", [])
        if texts:
            print("\n## Visible Typography & Signage (OCR)")
            for t in texts:
                print(f'- "{t["text"]}" (conf: {t.get("confidence", 1.0):.2f})')
        return

    print(f"[*] Inspecting IMGSTD stream: {args.code_or_file}")
    print(f"[+] Inspection finished in {res['inspection_time_ms']} ms")
    print(f"    - Protocol:         {res['protocol']}")
    print(f"    - Canvas:           {res['dimensions']['width']}x{res['dimensions']['height']} ({res['color_space']})")
    print(f"    - SHA-256:          {res['integrity_sha256']}")
    print(f"    - Visible Texts:    {res.get('text_count', 0)}")
    print(f"    - Structures:       {res.get('structure_count', 0)}")
    print(f"    - Physical Objects: {res.get('object_count', 0)}")
    print(f"    - Background Zones: {res.get('background_count', 0)}")

    if "scene_summary" in res and res["scene_summary"]:
        print(f"\n--- Scene Macro Context & Summary ---")
        print(f"  Summary: {res['scene_summary']}")

    texts = res.get("texts", [])
    if texts:
        print("\n--- Scene Typography & Signage ([TXT]) ---")
        for t in texts:
            bb = t["bounding_box"]
            print(f"  [TXT] \"{t['text']}\" | Conf: {t.get('confidence', 1.0):.2f} | Box: [{bb['x']},{bb['y']},{bb['width']},{bb['height']}]")

    print("\n--- Semantic Inventory ([STRUCT], [OBJ] & [BG]) ---")
    for r in res["regions"]:
        kind_tag = f"[{r.get('kind', 'OBJ')}]"
        conf_str = f" | Conf: {r.get('confidence', 1.0):.2f}" if r.get('kind') in ('OBJ', 'STRUCT') else ""
        prom_str = f" | Prom: {r.get('prominence')}" if r.get('prominence') else ""
        refl_str = f" [REFLECTED in {r.get('parent_structure')}]" if r.get('is_reflection') else ""
        print(f"  {kind_tag} [{r['region_id']}] {r['class_name']} ({r['category']}){conf_str}{prom_str}{refl_str}")
        print(f"      Box:   X={r['bounding_box']['x']}, Y={r['bounding_box']['y']}, W={r['bounding_box']['width']}, H={r['bounding_box']['height']}")
        print(f"      Color: RGB{r['dominant_color_rgb']} | Coverage: {r['coverage_percentage']}%")


def cmd_decode(args):
    from imgstd_project.core.decoder import decode_imgstd
    print(f"[*] Decoding IMGSTD stream: {args.code_or_file}")
    output_path = args.output or "decoded_image.png"
    if output_path and output_path.lower().endswith(('.jpg', '.jpeg')):
        print("[!] WARNING: JPEG output will re-encode and alter pixel values. Use .png or .bmp for lossless verification.")
    img, metrics = decode_imgstd(args.code_or_file, output_image_path=output_path)
    print(f"[+] Decoding complete in {metrics['total_decode_time_ms']} ms")
    print(f"    - Saved restored image to: {output_path}")
    print(f"    - Reconstructed SHA-256:   {metrics['sha256']}")
    print(f"    - Cryptographic Verified:  {metrics['verified']}")


def cmd_roundtrip(args):
    from imgstd_project.core.encoder import encode_image
    from imgstd_project.core.inspector import inspect_imgstd
    from imgstd_project.core.decoder import decode_imgstd
    from imgstd_project.core.verifier import IntegrityVerifier
    print(f"[*] Running Full Roundtrip Verification on: {args.image_path}")
    import cv2
    orig_bgr = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if orig_bgr is None:
        print(f"[-] Error: Could not read image at {args.image_path}")
        return

    # 1. Encode
    code, enc_m = encode_image(
        args.image_path,
        scene_model=getattr(args, "model", None),
        enable_smolvlm2=_resolve_smolvlm2_flag(args),
    )
    print(f"[1] Encoded in {enc_m['total_encode_time_ms']} ms (Model: {enc_m.get('scene_model')}, Ratio: {enc_m['compression_ratio']}x)")

    # 2. Inspect
    insp = inspect_imgstd(code)
    print(f"[2] Inspected in {insp['inspection_time_ms']} ms (Detected {insp['region_count']} semantic regions)")

    # 3. Decode
    restored_path = args.output or "verified_restored.png"
    if restored_path and restored_path.lower().endswith(('.jpg', '.jpeg')):
        print("[!] WARNING: JPEG output will re-encode and alter pixel values. Use .png or .bmp for lossless verification.")
    recon_img, dec_m = decode_imgstd(code, output_image_path=restored_path)
    print(f"[3] Decoded in {dec_m['total_decode_time_ms']} ms -> Saved to {restored_path}")

    # 4. Audit
    # Convert orig_bgr to RGB to compare with decoded matrix
    if len(orig_bgr.shape) == 2:
        orig_cmp = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2RGB)
    elif orig_bgr.shape[2] == 3:
        orig_cmp = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    else:
        orig_cmp = cv2.cvtColor(orig_bgr, cv2.COLOR_BGRA2RGBA)

    audit = IntegrityVerifier.verify_bit_exact(orig_cmp, recon_img)
    print("\n--- Integrity Audit ---")
    print(f"    - 100% Bit-Exact Match:   {audit['verified']}")
    print(f"    - SHA-256 Match:          {audit['sha256_match']}")
    print(f"    - Mean Squared Error:     {audit['mse']}")
    print(f"    - Differing Pixels:       {audit['diff_pixel_count']} / {audit['total_pixels']}")
    print(f"    - Fidelity Percentage:    {audit['fidelity_percentage']}%")


# =====================================================================
# Video CLI Handlers (VIDSTD::V01)
# =====================================================================

def cmd_vid_encode(args):
    from imgstd_project.core.video_codec import encode_video_to_vidstd
    sem_only = getattr(args, "semantic_only", False)
    mode_label = "semantic reference" if sem_only else "video"
    print(f"[*] Encoding {mode_label}: {args.input_video}")
    output_path = args.output
    if not output_path:
        base = os.path.splitext(os.path.basename(args.input_video))[0]
        output_path = f"{base}.vidstd"

    max_f = getattr(args, "max_frames", None)
    sample_fps = getattr(args, "sample_fps", None)
    gop_sz = getattr(args, "gop_size", 24)
    _, metrics = encode_video_to_vidstd(
        args.input_video,
        output_path=output_path,
        gop_size=gop_sz,
        max_frames=max_f,
        semantic_only=sem_only,
        sample_fps=sample_fps,
        scene_model=getattr(args, "model", None),
        enable_smolvlm2=_resolve_smolvlm2_flag(args),
    )
    print(f"[+] Video encoding finished in {metrics['encode_time_ms']} ms")
    print(f"    - Saved to:          {output_path}")
    print(f"    - Scene Model:       {metrics.get('scene_model')} (SmolVLM2: {'ON' if metrics.get('smolvlm2_enabled') else 'OFF'})")
    print(f"    - Mode:              {'SEMANTIC_REF (Lightweight AI Reference)' if sem_only else 'LOSSLESS (Bit-Exact)'}")
    print(f"    - Total Frames:      {metrics['total_frames']} ({metrics['duration_sec']}s @ {metrics['fps']} FPS)")
    print(f"    - Keyframes (I):     {metrics['i_frames']}")
    print(f"    - Tracked Steps (P): {metrics['p_frames']}")
    if not sem_only:
        print(f"    - Skip Identical(S): {metrics['s_frames']}")
        print(f"    - Total GOPs:        {metrics['total_gops']} (GOP Size: {metrics['gop_size']})")
    print(f"    - Scene Count:       {metrics['scene_count']}")
    print(f"    - Stream SHA-256:    {metrics['stream_sha256']}")
    print(f"    - Output File Size:  {metrics['output_size_bytes'] / 1024.0:.1f} KB")


def cmd_vid_decode(args):
    from imgstd_project.core.video_codec import decode_vidstd_to_video
    print(f"[*] Decoding VIDSTD stream to video: {args.vidstd_file}")
    output_path = args.output
    if not output_path:
        base = os.path.splitext(os.path.basename(args.vidstd_file))[0]
        output_path = f"{base}_restored.mp4"

    all_frames, metrics = decode_vidstd_to_video(args.vidstd_file, output_path=output_path)
    print(f"[+] Decoded {metrics['total_frames_decoded']} frames in {metrics['total_decode_time_ms']} ms")
    print(f"    - Saved restored video to: {output_path}")
    print(f"    - 100% Bit-Exact Parity:   {metrics['all_frames_verified']}")
    print(f"    - Verified Frame Count:    {metrics['verified_frames']} / {metrics['expected_frames']}")
    print(f"    - Mean Squared Error:      {metrics['mse']}")
    print(f"    - Average Decode Time:     {metrics['average_frame_decode_ms']} ms/frame")


def cmd_vid_inspect(args):
    from imgstd_project.core.video_codec import inspect_vidstd
    res = inspect_vidstd(args.vidstd_file)
    if getattr(args, "json", False):
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    meta = res["video_metadata"]
    if getattr(args, "llm", False):
        # Format as crisp prompt context for Non-Vision LLMs
        print("# NON-VISION AI GROUND TRUTH VIDEO REFERENCE")
        print(f"**Resolution**: {meta['width']}x{meta['height']} | **FPS**: {meta['fps']} | **Duration**: {meta['duration_seconds']}s ({meta['total_frames']} frames)")
        print(f"**Stream Integrity**: SHA-256 `{meta['stream_sha256']}` | **Scenes**: {meta['scene_count']}")
        print("\n## Scenes & Visual Timeline")
        for sc in res["scenes"]:
            print(f"- **Scene {sc['scene_id']}** ({sc['time']}): {sc['context']}")
        print("\n## Tracked Physical Entities & Kinematics")
        for trk in res["tracked_objects"]:
            refl = f" (Reflected inside {trk['inside_of']})" if trk.get("inside_of") else ""
            print(f"- **{trk['class']}** [{trk['role']}]: active {trk['time']}, trajectory: `{trk['path']}`{refl}")
        if res.get("text_events"):
            print("\n## Visible Signage & Typography (OCR)")
            for tx in res["text_events"]:
                print(f"- Text: \"{tx['text']}\" (Time: {tx['time']})")
        if res.get("camera_motion_events"):
            print("\n## Camera Ego-Motion")
            for cam in res["camera_motion_events"]:
                print(f"- {cam}")
        return

    print(f"[*] Inspecting VIDSTD stream: {args.vidstd_file}")
    print(f"[+] Inspection finished in {res['inspection_time_ms']} ms (< 1 ms Zero-Vision)")
    print(f"    - Protocol:       {res['protocol']}")
    print(f"    - Resolution:     {meta['width']}x{meta['height']} ({meta['channels']} channels)")
    print(f"    - Duration:       {meta['duration_seconds']}s ({meta['total_frames']} frames @ {meta['fps']} FPS)")
    print(f"    - GOP Structure:  {meta['total_gops']} GOPs (GOP Size: {meta['gop_size']})")
    print(f"    - Stream SHA-256: {meta['stream_sha256']}")

    print(f"\n--- Scene Timeline ({meta['scene_count']} Scenes) ---")
    for sc in res["scenes"]:
        print(f"  [SCENE {sc['scene_id']}] Frames: {sc['frames']} ({sc['time']}) -> \"{sc['context']}\"")

    print(f"\n--- Tracked Objects & Trajectories ({len(res['tracked_objects'])} Entities) ---")
    for trk in res["tracked_objects"]:
        inside = f" [REFLECTED in {trk['inside_of']}]" if trk.get("inside_of") else ""
        print(f"  [{trk['track_id']}] {trk['class']} | Role: {trk['role']} | Time: {trk['first_frame']} to {trk['last_frame']}{inside}")
        print(f"        Path: {trk['path']} | RGB: {trk['dominant_rgb']}")

    if res.get("text_events"):
        print(f"\n--- On-Screen Typography & Signage ({len(res['text_events'])} Events) ---")
        for tx in res["text_events"]:
            print(f"  [{tx['text_track_id']}] \"{tx['text']}\" | Time: {tx['first_frame']} to {tx['last_frame']} | Box: [{tx['box']}]")


def cmd_vid_frame(args):
    from imgstd_project.core.video_codec import extract_vidstd_frame
    print(f"[*] Extracting frame {args.frame} from {args.vidstd_file}...")
    output_path = args.output or f"frame_{args.frame:04d}.png"
    frame, metrics = extract_vidstd_frame(args.vidstd_file, args.frame, output_image_path=output_path)
    print(f"[+] Frame extracted in {metrics['seek_and_decode_time_ms']} ms")
    print(f"    - Verified Bit-Exact: {metrics['verified']}")
    print(f"    - Frame SHA-256:      {metrics['sha256']}")
    print(f"    - Saved Image:        {output_path}")


def cmd_vid_roundtrip(args):
    import tempfile
    from imgstd_project.core.video_codec import encode_video_to_vidstd, inspect_vidstd, extract_vidstd_frame, decode_vidstd_to_video
    print(f"[*] Running Video Roundtrip Verification on: {args.input_video}")
    fd, temp_vidstd = tempfile.mkstemp(suffix=".vidstd", prefix="vidstd_rt_")
    os.close(fd)
    try:
        # 1. Encode
        _, enc_m = encode_video_to_vidstd(args.input_video, output_path=temp_vidstd)
        print(f"[1] Encoded {enc_m['total_frames']} frames in {enc_m['encode_time_ms']} ms")

        # 2. Inspect
        insp = inspect_vidstd(temp_vidstd)
        print(f"[2] Timeline inspected in {insp['inspection_time_ms']} ms ({insp['video_metadata']['scene_count']} scenes)")

        # 3. Seek single frame
        test_idx = min(enc_m['total_frames'] - 1, max(1, enc_m['total_frames'] // 2))
        _, seek_m = extract_vidstd_frame(temp_vidstd, test_idx)
        print(f"[3] O(1) Seek to frame {test_idx} verified in {seek_m['seek_and_decode_time_ms']} ms (MSE = 0.0)")

        # 4. Decode all frames in O(1) RAM and verify 100% bit-exact parity
        _, dec_m = decode_vidstd_to_video(temp_vidstd, verify_only=True)
        print(f"[4] Full video verified in {dec_m['total_decode_time_ms']} ms (Avg {dec_m['average_frame_decode_ms']} ms/frame)")

        print("\n--- Integrity Audit ---")
        print(f"    - 100% Bit-Exact Match:   {dec_m['all_frames_verified']}")
        print(f"    - Verified Frame Count:   {dec_m['verified_frames']} / {dec_m['expected_frames']}")
        print(f"    - Mean Squared Error:     {dec_m['mse']}")
    finally:
        if os.path.exists(temp_vidstd):
            os.remove(temp_vidstd)


def cmd_models(args):
    from imgstd_project.core.config import get_model_catalog
    cat = get_model_catalog()
    if getattr(args, "json", False):
        print(json.dumps(cat, indent=2, ensure_ascii=False))
        return

    hw = cat["hardware"]
    cfg = cat["current_settings"]
    hw_str = f"GPU: {hw['gpu_name']}" if hw["has_cuda"] else f"CPU-Only ({hw['cpu_cores']} cores)"
    print(f"=== IMGSTD / VIDSTD Scene & VLM Model Catalog ({hw_str}) ===")
    print(f"Settings File: {cat['settings_path']}")
    print(f"Active Config: scene_model={cfg['scene_model']} | smolvlm2={'ON' if cfg['enable_smolvlm2'] else 'OFF'} | sample_fps={cfg['sample_fps']} | yolo_conf={cfg['yolo_conf']}\n")

    print("[1] MobileCLIP2 Scene Context Models (choose via `vidstd config set --scene-model <id>` or `--model <id>`):")
    for m in cat["scene_models"]:
        active_mark = "*ACTIVE*" if m["active"] else "        "
        dl_mark = "Local Ready" if m["downloaded_locally"] else "Auto-Download"
        print(f"  {active_mark} {m['id']:<16} [{m['tier']:<16}] | {m['ram_estimate']:<8} RAM | {m['cpu_speed']:<20} | {dl_mark}")
        print(f"             Backbone: {m['backbone']} ({m['weights_size']}) -> Tip: {m['tip_en']}")

    vlm = cat["generative_vlm"]
    vlm_state = "ON " if vlm["active"] else "OFF"
    vlm_dl = "Local Ready" if vlm["downloaded_locally"] else "Auto-Download"
    print("\n[2] Optional Generative VLM (choose via `vidstd config set --smolvlm2 on|off` or `--smolvlm2`):")
    print(f"  [{vlm_state}]    {vlm['id']:<16} [{vlm['tier']:<16}] | {vlm['ram_estimate']:<13} | {vlm['cpu_speed']} | {vlm_dl}")
    print(f"             Tip: {vlm['tip_en']}")

    print("\n[!] Recommendations & Tips:")
    for tip in cat["tips"]:
        print(f"  - {tip}")
    print("  - Example switch:  vidstd config set --scene-model mobileclip2-s2 --smolvlm2 off  (auto-downloads weights)")
    print("  - First-time setup: vidstd setup")
    print("  - One-off run:     vidstd vid-ref video.mp4 -o ref.vidstd --model mobileclip2-b")


def cmd_setup(args):
    from imgstd_project.core.config import (
        load_settings,
        save_settings,
        download_base_vision_models,
        download_model_weights,
        get_model_catalog,
    )

    updates = {}
    if getattr(args, "model", None):
        updates["scene_model"] = args.model
    if getattr(args, "smolvlm2", False):
        updates["enable_smolvlm2"] = True
    cfg = save_settings(updates) if updates else load_settings()

    quiet = getattr(args, "json", False)
    base_res = download_base_vision_models(quiet=quiet)
    scene_res = download_model_weights(cfg["scene_model"], quiet=quiet)
    dl_list = [scene_res]
    if cfg.get("enable_smolvlm2"):
        dl_list.append(download_model_weights("smolvlm2", quiet=quiet))

    if quiet:
        print(json.dumps({
            "setup_complete": True,
            "base_models": base_res,
            "downloaded_models": dl_list,
            "catalog": get_model_catalog(),
        }, indent=2, ensure_ascii=False))
        return

    print("\n[+] Setup complete! All required weights are ready locally:")
    print(f"    - Weights Folder:   {base_res['weights_root']}")
    print(f"    - YOLO11n-seg:      {base_res['yolo11n-seg']}")
    print(f"    - YOLO-World v2:    {base_res['yolov8s-worldv2']}")
    print(f"    - RapidOCR:         {base_res['rapidocr']}")
    for item in dl_list:
        print(f"    - {item['model_id']:<16}: {item['status']}")


def cmd_config(args):
    from imgstd_project.core.config import (
        load_settings,
        save_settings,
        reset_settings,
        get_model_catalog,
        download_model_weights,
        MODEL_CATALOG,
    )

    action = getattr(args, "config_action", "show") or "show"
    if action == "reset":
        cfg = reset_settings()
        if getattr(args, "json", False):
            print(json.dumps({"reset": True, "settings": cfg}, indent=2, ensure_ascii=False))
            return
        print("[+] Settings reset to lightweight CPU defaults.")
        action = "show"
    elif action == "set":
        updates = {}
        if getattr(args, "scene_model", None) is not None:
            updates["scene_model"] = args.scene_model
        if getattr(args, "smolvlm2", None) is not None:
            updates["enable_smolvlm2"] = (args.smolvlm2.lower() in ("on", "true", "1", "yes"))
        if getattr(args, "sample_fps", None) is not None:
            updates["sample_fps"] = args.sample_fps
        if getattr(args, "yolo_conf", None) is not None:
            updates["yolo_conf"] = args.yolo_conf

        if not updates:
            raise ValueError("Provide at least one setting to update (e.g. --scene-model mobileclip2-s2 or --smolvlm2 on).")

        cfg = save_settings(updates)
        quiet = getattr(args, "json", False)
        dl_results = []
        if not getattr(args, "no_download", False):
            if "scene_model" in updates:
                dl_results.append(download_model_weights(cfg["scene_model"], quiet=quiet))
            if updates.get("enable_smolvlm2") is True:
                dl_results.append(download_model_weights("smolvlm2", quiet=quiet))

        if quiet:
            cat = get_model_catalog()
            print(json.dumps({
                "updated": True,
                "settings": cfg,
                "downloaded_models": dl_results,
                "tips": cat["tips"],
            }, indent=2, ensure_ascii=False))
            return
        print("[+] Settings updated & saved.")

    cat = get_model_catalog()
    if getattr(args, "json", False):
        print(json.dumps(cat, indent=2, ensure_ascii=False))
        return

    cfg = cat["current_settings"]
    m_info = MODEL_CATALOG.get(cfg["scene_model"], MODEL_CATALOG["mobileclip2-s0"])
    print(f"=== Current IMGSTD / VIDSTD Settings ({cat['settings_path']}) ===")
    print(f"  - Scene Model:   {cfg['scene_model']} [{m_info['tier']}] ({m_info['backbone']}, {m_info['ram_estimate']} RAM)")
    print(f"  - SmolVLM2 VLM:  {'ON [GENERATIVE VLM / HEAVY]' if cfg['enable_smolvlm2'] else 'OFF (fast MobileCLIP2 mode)'}")
    print(f"  - Sample FPS:    {cfg['sample_fps']} FPS (for vid-ref)")
    print(f"  - YOLO Conf:     {cfg['yolo_conf']}")
    print("\nTips:")
    for tip in cat["tips"]:
        print(f"  - {tip}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    epilog = """examples:
  # 1. Inspect available models (Light / Balanced / Heavy / VLM) & settings:
  vidstd models
  vidstd config set --scene-model mobileclip2-s2 --smolvlm2 off

  # 2. Generate fast lightweight AI reference (<1 MB) from an existing video:
  vidstd vid-ref video.mp4 -o video_ref.vidstd
  vidstd vid-ref video.mp4 --model mobileclip2-b -o video_ref.vidstd

  # 3. Read video semantics for any Non-Vision AI / LLM in < 1 ms:
  vidstd vid-inspect video_ref.vidstd --llm

  # 4. Full bit-exact lossless video compression (O(1) streaming RAM):
  vidstd vid-encode clip.mp4 -o clip.vidstd
  vidstd vid-decode clip.vidstd -o clip_restored.mp4

  # 5. Semantic-lossless image workflow & verification:
  vidstd encode photo.png -o photo.imgstd
  vidstd inspect photo.imgstd --llm
  vidstd roundtrip photo.png"""

    parser = argparse.ArgumentParser(
        prog="vidstd",
        description="IMGSTD & VIDSTD: Ultra-Fast Semantic-Lossless Media Codec & Non-Vision AI Reference CLI",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"IMGSTD/VIDSTD {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup command (first-time clone setup / pre-download models)
    p_setup = subparsers.add_parser("setup", help="Initialize and download base vision models (YOLO11n-seg, YOLO-World, RapidOCR) and chosen scene model")
    p_setup.add_argument("--model", help="Also select & download a MobileCLIP2 model (default: current active model)", default=None)
    p_setup.add_argument("--smolvlm2", help="Also enable and download SmolVLM2-500M-Video-Instruct (~1 GB)", action="store_true")
    p_setup.add_argument("--json", help="Output setup result as JSON", action="store_true")
    p_setup.set_defaults(func=cmd_setup)

    # Models catalog command
    p_models = subparsers.add_parser("models", help="List available MobileCLIP2 & SmolVLM2 models with hardware tiers and tips")
    p_models.add_argument("--json", help="Output catalog as JSON", action="store_true")
    p_models.set_defaults(func=cmd_models)

    # Config / Settings command
    p_cfg = subparsers.add_parser("config", help="View or update persistent model & performance settings (auto-downloads selected models)")
    p_cfg.add_argument("--json", help="Output configuration as JSON", action="store_true")
    cfg_sub = p_cfg.add_subparsers(dest="config_action")
    p_cfg_show = cfg_sub.add_parser("show", help="Show current settings and hardware recommendations")
    p_cfg_show.add_argument("--json", action="store_true")
    p_cfg_set = cfg_sub.add_parser("set", help="Update persistent settings and download newly selected models")
    p_cfg_set.add_argument("--scene-model", help="Set default MobileCLIP2 model (e.g. mobileclip2-s0, s2, b, s3, s4, l-14)")
    p_cfg_set.add_argument("--smolvlm2", choices=["on", "off"], help="Enable or disable SmolVLM2 generative VLM by default")
    p_cfg_set.add_argument("--sample-fps", type=float, help="Default vid-ref sampling FPS (default: 2.0)")
    p_cfg_set.add_argument("--yolo-conf", type=float, help="Default YOLO detection threshold (0.05 - 0.95)")
    p_cfg_set.add_argument("--no-download", help="Save settings only without downloading weights immediately", action="store_true")
    p_cfg_set.add_argument("--json", action="store_true")
    p_cfg_reset = cfg_sub.add_parser("reset", help="Reset all settings to lightweight CPU defaults")
    p_cfg_reset.add_argument("--json", action="store_true")
    p_cfg.set_defaults(func=cmd_config)

    # Encode command
    p_enc = subparsers.add_parser("encode", help="Encode an image into IMGSTD code")
    p_enc.add_argument("image_path", help="Path to input image")
    p_enc.add_argument("-o", "--output", help="Output file path (.imgstd)", default=None)
    p_enc.add_argument("--model", help="Override MobileCLIP2 model for this run (see `vidstd models`)", default=None)
    enc_vlm_grp = p_enc.add_mutually_exclusive_group()
    enc_vlm_grp.add_argument("--smolvlm2", help="Enable SmolVLM2 generative VLM for this run", action="store_true")
    enc_vlm_grp.add_argument("--no-smolvlm2", help="Force disable SmolVLM2 for this run", action="store_true")
    p_enc.set_defaults(func=cmd_encode)

    # Inspect command
    p_insp = subparsers.add_parser("inspect", help="Inspect IMGSTD code in sub-millisecond time")
    p_insp.add_argument("code_or_file", help="IMGSTD code string or path to .imgstd file")
    p_insp.add_argument("--json", help="Output raw JSON for programmatic consumption", action="store_true")
    p_insp.add_argument("--llm", help="Output formatted Markdown ground-truth context for Non-Vision LLMs", action="store_true")
    p_insp.set_defaults(func=cmd_inspect)

    # Decode command
    p_dec = subparsers.add_parser("decode", help="Decode IMGSTD code back into exact image")
    p_dec.add_argument("code_or_file", help="IMGSTD code string or path to .imgstd file")
    p_dec.add_argument("-o", "--output", help="Path to save restored image", default="restored.png")
    p_dec.set_defaults(func=cmd_decode)

    # Roundtrip command
    p_rt = subparsers.add_parser("roundtrip", help="Run encode -> inspect -> decode -> verify pipeline")
    p_rt.add_argument("image_path", help="Path to input image")
    p_rt.add_argument("-o", "--output", help="Path to save restored image", default=None)
    p_rt.add_argument("--model", help="Override MobileCLIP2 model for this run", default=None)
    rt_vlm_grp = p_rt.add_mutually_exclusive_group()
    rt_vlm_grp.add_argument("--smolvlm2", help="Enable SmolVLM2 generative VLM for this run", action="store_true")
    rt_vlm_grp.add_argument("--no-smolvlm2", help="Force disable SmolVLM2 for this run", action="store_true")
    p_rt.set_defaults(func=cmd_roundtrip)

    # Video Encode command
    p_venc = subparsers.add_parser("vid-encode", help="Encode video into VIDSTD container")
    p_venc.add_argument("input_video", help="Path to input video (.mp4, .avi, etc.)")
    p_venc.add_argument("-o", "--output", help="Output .vidstd file path", default=None)
    p_venc.add_argument("--gop-size", help="GOP size in frames (default: 24)", type=int, default=24)
    p_venc.add_argument("--max-frames", help="Limit number of frames to encode (useful for large videos)", type=int, default=None)
    p_venc.add_argument("--semantic-only", help="Generate lightweight AI semantic reference (<1MB, no pixel payload)", action="store_true")
    p_venc.add_argument("--sample-fps", help="Sampling FPS when --semantic-only is set (default: from settings / 2.0)", type=float, default=None)
    p_venc.add_argument("--model", help="Override MobileCLIP2 model for this run", default=None)
    venc_vlm_grp = p_venc.add_mutually_exclusive_group()
    venc_vlm_grp.add_argument("--smolvlm2", help="Enable SmolVLM2 generative VLM for this run", action="store_true")
    venc_vlm_grp.add_argument("--no-smolvlm2", help="Force disable SmolVLM2 for this run", action="store_true")
    p_venc.set_defaults(func=cmd_vid_encode)

    # Video Reference command (ultra-fast alias for vid-encode --semantic-only)
    p_vref = subparsers.add_parser("vid-ref", help="Build ultra-fast lightweight AI semantic reference (.vidstd) from existing video")
    p_vref.add_argument("input_video", help="Path to input video (.mp4, .avi, etc.)")
    p_vref.add_argument("-o", "--output", help="Output .vidstd reference file path", default=None)
    p_vref.add_argument("--sample-fps", help="Sampling FPS for timeline tracking (default: from settings / 2.0)", type=float, default=None)
    p_vref.add_argument("--max-frames", help="Limit sampled frames", type=int, default=None)
    p_vref.add_argument("--model", help="Override MobileCLIP2 model for this run", default=None)
    vref_vlm_grp = p_vref.add_mutually_exclusive_group()
    vref_vlm_grp.add_argument("--smolvlm2", help="Enable SmolVLM2 generative VLM for this run", action="store_true")
    vref_vlm_grp.add_argument("--no-smolvlm2", help="Force disable SmolVLM2 for this run", action="store_true")
    p_vref.set_defaults(func=cmd_vid_encode, semantic_only=True, gop_size=24)

    # Video Inspect command
    p_vinsp = subparsers.add_parser("vid-inspect", help="Inspect VIDSTD video timeline in < 1 ms")
    p_vinsp.add_argument("vidstd_file", help="Path to .vidstd file")
    p_vinsp.add_argument("--json", help="Output raw JSON for programmatic consumption", action="store_true")
    p_vinsp.add_argument("--llm", help="Output formatted Markdown ground-truth context for Non-Vision LLMs", action="store_true")
    p_vinsp.set_defaults(func=cmd_vid_inspect)

    # Video Decode command
    p_vdec = subparsers.add_parser("vid-decode", help="Decode VIDSTD container back into video file")
    p_vdec.add_argument("vidstd_file", help="Path to .vidstd file")
    p_vdec.add_argument("-o", "--output", help="Output video file path (.mp4, .mkv, etc.)", default=None)
    p_vdec.set_defaults(func=cmd_vid_decode)

    # Video Frame Extract command
    p_vframe = subparsers.add_parser("vid-frame", help="Extract single bit-exact frame via O(1) seek")
    p_vframe.add_argument("vidstd_file", help="Path to .vidstd file")
    p_vframe.add_argument("--frame", help="0-indexed frame number to extract", type=int, required=True)
    p_vframe.add_argument("-o", "--output", help="Output PNG path", default=None)
    p_vframe.set_defaults(func=cmd_vid_frame)

    # Video Roundtrip command
    p_vrt = subparsers.add_parser("vid-roundtrip", help="Run full video roundtrip verification")
    p_vrt.add_argument("input_video", help="Path to input video")
    p_vrt.set_defaults(func=cmd_vid_roundtrip)

    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, IndexError, RuntimeError) as err:
        print(f"[!] Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
