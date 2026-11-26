"""
IMGSTD / VIDSTD Persistent Configuration & Model Catalog Registry.
Manages user & AI preferences (active MobileCLIP2 variant, SmolVLM2 generative toggle,
sampling FPS, YOLO confidence) along with hardware-aware recommendations and practical tips.
"""

import os
import json
from typing import Dict, Any, Optional, List


DEFAULT_SETTINGS: Dict[str, Any] = {
    "scene_model": "mobileclip2-s0",
    "enable_smolvlm2": False,
    "sample_fps": 2.0,
    "yolo_conf": 0.40,
}


# Ordered from lightest/fastest (default CPU) to heaviest/flagship
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "mobileclip2-s0": {
        "id": "mobileclip2-s0",
        "open_clip_name": "MobileCLIP2-S0",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "LIGHTWEIGHT",
        "tier_label": "خفيف جداً (موصى به للأجهزة الضعيفة والـ CPU)",
        "backbone": "FastViT-MCi0 (256px)",
        "weights_size": "~44 MB visual",
        "ram_estimate": "~93 MB",
        "cpu_speed": "~75 ms / frame",
        "gpu_recommended": False,
        "tip": "الافتراضي والأسرع: يعطي دقة مشهد ممتازة مع أقل استهلاك للرامات، ومناسب جداً للفيديوهات الطويلة على CPU.",
        "tip_en": "Default & fastest: low RAM (~93 MB), bundled locally, ideal for CPU-only devices and long videos.",
    },
    "mobileclip2-s2": {
        "id": "mobileclip2-s2",
        "open_clip_name": "MobileCLIP2-S2",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "BALANCED",
        "tier_label": "متوازن (دقة أعلى على المعالج)",
        "backbone": "FastViT-MCi2 (256px)",
        "weights_size": "~140 MB",
        "ram_estimate": "~220 MB",
        "cpu_speed": "~120 ms / frame",
        "gpu_recommended": False,
        "tip": "ترقية متوازنة للـ CPU تعطي فهماً أدق للتفاصيل مع سرعة قريبة من S0.",
        "tip_en": "Balanced CPU upgrade: sharper semantic accuracy while remaining fast on CPUs.",
    },
    "mobileclip2-b": {
        "id": "mobileclip2-b",
        "open_clip_name": "MobileCLIP2-B",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "BALANCED-VIT",
        "tier_label": "متوازن-متوسط (Vision Transformer)",
        "backbone": "ViT-Base-MCi (224px)",
        "weights_size": "~340 MB",
        "ram_estimate": "~450 MB",
        "cpu_speed": "~200 ms / frame",
        "gpu_recommended": False,
        "tip": "مبني على ViT-Base؛ ممتاز في تمييز الأشياء الدقيقة والمشاهد المعقدة.",
        "tip_en": "ViT-Base architecture; strong fine-grained scene and object context.",
    },
    "mobileclip2-s3": {
        "id": "mobileclip2-s3",
        "open_clip_name": "MobileCLIP2-S3",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "HIGH-ACCURACY",
        "tier_label": "عالي الدقة (أثقل على CPU)",
        "backbone": "FastViT-MCi3 (256px, 768-d)",
        "weights_size": "~490 MB",
        "ram_estimate": "~650 MB",
        "cpu_speed": "~280 ms / frame",
        "gpu_recommended": True,
        "tip": "مناسب إذا كان لديك 8GB+ RAM أو GPU وتريد دقة تصنيف أعلى للمشاهد المتشابهة.",
        "tip_en": "Rich 768-d embeddings for subtle scenes; best with 8+ GB RAM or GPU.",
    },
    "mobileclip2-s4": {
        "id": "mobileclip2-s4",
        "open_clip_name": "MobileCLIP2-S4",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "HEAVY",
        "tier_label": "ثقيل (أعلى سعة FastViT)",
        "backbone": "FastViT-MCi4 (256px, 768-d)",
        "weights_size": "~900 MB",
        "ram_estimate": "~1.2 GB",
        "cpu_speed": "~450 ms / frame",
        "gpu_recommended": True,
        "tip": "ثقيل نسبياً على المعالج الضعيف؛ يُنصح به في تحليل الصور أو عند توفر GPU.",
        "tip_en": "Heavyweight FastViT; best for single images or GPU-accelerated video runs.",
    },
    "mobileclip2-l-14": {
        "id": "mobileclip2-l-14",
        "open_clip_name": "MobileCLIP2-L-14",
        "pretrained_tag": "dfndr2b",
        "kind": "scene_clip",
        "tier": "FLAGSHIP / HEAVY",
        "tier_label": "ثقيل جداً / الرائد (أقصى دقة CLIP)",
        "backbone": "ViT-Large-14 (224px, 768-d)",
        "weights_size": "~1.2 GB",
        "ram_estimate": "~1.8 GB",
        "cpu_speed": "~800+ ms / frame",
        "gpu_recommended": True,
        "tip": "أعلى دقة بين جميع نسخ MobileCLIP2، لكنه بطيء على CPU — يفضّل GPU أو عدد إطارات محدود.",
        "tip_en": "Highest accuracy CLIP variant, slow on CPU — strongly recommended with GPU.",
    },
}


SMOLVLM2_INFO: Dict[str, Any] = {
    "id": "smolvlm2",
    "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "revision": "main",
    "kind": "generative_vlm",
    "tier": "GENERATIVE VLM / HEAVY",
    "tier_label": "نموذج رؤية ولغة توليدي ثقيل (اختياري)",
    "backbone": "SmolVLM2-500M-Video-Instruct",
    "weights_size": "~1.0 GB",
    "ram_estimate": "~2.2 - 3.0 GB",
    "cpu_speed": "~10 - 35 sec / keyframe (CPU) | ~0.5 sec (GPU)",
    "gpu_recommended": True,
    "tip": "يولّد وصفاً نصياً حراً ومفصلاً للمشاهد والأحداث الزمنية، لكنه بطيء جداً على CPU. فعّله للصور المهمة أو مع GPU فقط.",
    "tip_en": "Generates rich open-ended scene & temporal clip descriptions. Very slow on CPU; recommended for GPU or single key images.",
}


def get_settings_path() -> str:
    """Returns persistent settings JSON path (~/.imgstd/settings.json or IMGSTD_CONFIG_PATH)."""
    env_path = os.environ.get("IMGSTD_CONFIG_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(os.path.expanduser("~"), ".imgstd", "settings.json")


def normalize_scene_model_name(name: Optional[str]) -> str:
    """Normalizes user-supplied MobileCLIP2 alias into canonical catalog ID."""
    if not name:
        return DEFAULT_SETTINGS["scene_model"]
    raw = str(name).strip().lower().replace("_", "-")
    aliases = {
        "s0": "mobileclip2-s0",
        "s2": "mobileclip2-s2",
        "s3": "mobileclip2-s3",
        "s4": "mobileclip2-s4",
        "b": "mobileclip2-b",
        "l14": "mobileclip2-l-14",
        "l-14": "mobileclip2-l-14",
        "mobileclip2-l14": "mobileclip2-l-14",
    }
    canon = aliases.get(raw, raw)
    if canon not in MODEL_CATALOG:
        valid = ", ".join(MODEL_CATALOG.keys())
        raise ValueError(f"Unknown scene model '{name}'. Valid options: {valid}")
    return canon


def _package_weights_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights"))


def get_weights_root() -> str:
    """
    Resolves a writable cross-platform weights directory for downloaded models:
    1. IMGSTD_WEIGHTS_DIR environment variable if set.
    2. Repo-local `<project_root>/weights` if running from a writable source checkout.
    3. User home fallback `~/.imgstd/weights` (when installed into read-only site-packages).
    """
    env_dir = os.environ.get("IMGSTD_WEIGHTS_DIR")
    if env_dir:
        target = os.path.abspath(env_dir)
        os.makedirs(target, exist_ok=True)
        return target

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pkg_w = os.path.join(repo_root, "weights")
    if os.path.isdir(pkg_w) or os.path.exists(os.path.join(repo_root, "pyproject.toml")):
        try:
            os.makedirs(pkg_w, exist_ok=True)
            return pkg_w
        except OSError:
            pass

    home_w = os.path.join(os.path.expanduser("~"), ".imgstd", "weights")
    os.makedirs(home_w, exist_ok=True)
    return home_w


def _unique_weight_bases() -> List[str]:
    candidates = [
        os.path.join(os.getcwd(), "weights"),
        _package_weights_root(),
        get_weights_root(),
        os.path.join(os.path.expanduser("~"), ".imgstd", "weights"),
    ]
    seen = set()
    out: List[str] = []
    for c in candidates:
        ab = os.path.abspath(c)
        if ab not in seen:
            seen.add(ab)
            out.append(ab)
    return out


def is_model_downloaded_locally(model_id: str) -> bool:
    """Checks if model weights are already present in local weights/ folder or HF/OpenCLIP cache."""
    if model_id == "smolvlm2":
        for base in _unique_weight_bases():
            if os.path.exists(os.path.join(base, "SmolVLM2-500M-Video-Instruct", "model.safetensors")):
                return True
        hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.isdir(hf_hub):
            for entry in os.listdir(hf_hub):
                if entry.lower() == "models--huggingfacetb--smolvlm2-500m-video-instruct":
                    return True
        return False

    canon = MODEL_CATALOG[model_id]["open_clip_name"]
    for base in _unique_weight_bases():
        if os.path.exists(os.path.join(base, canon, "open_clip_model.safetensors")):
            return True
    hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_hub):
        prefix = f"models--timm--{canon}-OpenCLIP".lower()
        for entry in os.listdir(hf_hub):
            if entry.lower() == prefix:
                return True
    return False


def download_model_weights(model_id: str, quiet: bool = False) -> Dict[str, Any]:
    """
    Downloads and caches weights for a selected model ('mobileclip2-*' or 'smolvlm2')
    so switching models in settings automatically makes them ready on any cloned machine.
    """
    import time

    t0 = time.perf_counter()
    if model_id == "smolvlm2":
        if is_model_downloaded_locally("smolvlm2"):
            return {"model_id": "smolvlm2", "status": "already_ready", "downloaded": False, "elapsed_ms": 0.0}
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLM2 requires 'huggingface_hub', 'transformers', and 'accelerate'. "
                "Install with: pip install -e \".[neural]\" (or pip install transformers huggingface_hub accelerate)."
            ) from exc

        target_dir = os.path.join(get_weights_root(), "SmolVLM2-500M-Video-Instruct")
        os.makedirs(target_dir, exist_ok=True)
        if not quiet:
            print(f"[*] Downloading SmolVLM2 ({SMOLVLM2_INFO['model_id']}) to {target_dir}...")
        snapshot_download(
            repo_id=SMOLVLM2_INFO["model_id"],
            revision=SMOLVLM2_INFO.get("revision", "main"),
            local_dir=target_dir,
            ignore_patterns=["*.onnx", "onnx/*", "*.bin"],
        )
        elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
        if not quiet:
            print(f"[+] SmolVLM2 downloaded and ready ({elapsed} ms).")
        return {"model_id": "smolvlm2", "status": "downloaded", "downloaded": True, "path": target_dir, "elapsed_ms": elapsed}

    canon_id = normalize_scene_model_name(model_id)
    meta = MODEL_CATALOG[canon_id]
    was_cached = is_model_downloaded_locally(canon_id)
    cache_dir = os.path.join(get_weights_root(), meta["open_clip_name"])
    cache_file = os.path.join(cache_dir, "prompt_bank_v1.npz")
    ts_file = os.path.join(cache_dir, "visual_model.ts")
    onnx_file = os.path.join(cache_dir, "visual_model.onnx")

    from .scene_clip import get_mobileclip_analyzer
    analyzer = get_mobileclip_analyzer(canon_id)
    if was_cached and analyzer._load_prompt_bank_only(cache_file) and (os.path.exists(onnx_file) or os.path.exists(ts_file)):
        return {"model_id": canon_id, "status": "already_ready", "downloaded": False, "path": cache_file, "elapsed_ms": 0.0}

    if not quiet:
        msg = "Building prompt bank cache" if was_cached else f"Downloading {meta['open_clip_name']} ({meta['weights_size']})"
        print(f"[*] {msg} for '{canon_id}'...")

    ok = analyzer._ensure_loaded()
    if not ok:
        raise RuntimeError(
            f"Failed to download/initialize '{canon_id}'. Check internet connection and open_clip_torch installation."
        )
    elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
    if not quiet:
        print(f"[+] '{canon_id}' is ready ({elapsed} ms).")
    return {
        "model_id": canon_id,
        "status": "cached" if was_cached else "downloaded",
        "downloaded": not was_cached,
        "path": analyzer._get_cache_dir(),
        "elapsed_ms": elapsed,
    }


def download_base_vision_models(quiet: bool = False) -> Dict[str, Any]:
    """
    Ensures core object/segmentation/OCR models (YOLO11n-seg, YOLO-World, RapidOCR)
    are downloaded into the active weights directory for offline-ready runs.
    """
    import time
    t0 = time.perf_counter()
    results: Dict[str, Any] = {}
    w_root = get_weights_root()
    if not quiet:
        print(f"[*] Initializing core vision models in: {w_root}")

    from .extractor import (
        get_yolo_model,
        get_hybrid_yolo_world_ort,
        get_yolo_world_model,
        _set_world_classes_cached,
        YoloWorldSemanticExtractor,
    )
    from .ocr import get_ocr_engine

    yolo = get_yolo_model("yolo11n-seg.pt")
    results["yolo11n-seg"] = "ready" if yolo is not None else "unavailable"
    world_ort = get_hybrid_yolo_world_ort(YoloWorldSemanticExtractor.HYBRID_OPEN_CLASSES, "yolov8s-worldv2.pt")
    if world_ort is not None:
        results["yolov8s-worldv2"] = "ready (onnx)"
    else:
        world = get_yolo_world_model("yolov8s-worldv2.pt")
        if world is not None:
            try:
                _set_world_classes_cached(world, YoloWorldSemanticExtractor.HYBRID_OPEN_CLASSES)
            except Exception:
                pass
        results["yolov8s-worldv2"] = "ready" if world is not None else "unavailable"
    ocr = get_ocr_engine()
    results["rapidocr"] = "ready" if ocr is not None else "unavailable"

    results["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    results["weights_root"] = w_root
    return results


def load_settings() -> Dict[str, Any]:
    """Loads settings cleanly, falling back to safe lightweight defaults."""
    cfg = dict(DEFAULT_SETTINGS)
    path = get_settings_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "scene_model" in data:
                    try:
                        cfg["scene_model"] = normalize_scene_model_name(data["scene_model"])
                    except ValueError:
                        pass
                if "enable_smolvlm2" in data:
                    cfg["enable_smolvlm2"] = bool(data["enable_smolvlm2"])
                if "sample_fps" in data:
                    try:
                        val = float(data["sample_fps"])
                        if val > 0:
                            cfg["sample_fps"] = val
                    except (TypeError, ValueError):
                        pass
                if "yolo_conf" in data:
                    try:
                        val = float(data["yolo_conf"])
                        if 0.05 <= val <= 0.95:
                            cfg["yolo_conf"] = val
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    return cfg


def save_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Validates, merges, and persists settings to ~/.imgstd/settings.json."""
    cfg = load_settings()
    if "scene_model" in updates and updates["scene_model"] is not None:
        cfg["scene_model"] = normalize_scene_model_name(updates["scene_model"])
    if "enable_smolvlm2" in updates and updates["enable_smolvlm2"] is not None:
        cfg["enable_smolvlm2"] = bool(updates["enable_smolvlm2"])
    if "sample_fps" in updates and updates["sample_fps"] is not None:
        val = float(updates["sample_fps"])
        if val <= 0:
            raise ValueError("sample_fps must be > 0.")
        cfg["sample_fps"] = val
    if "yolo_conf" in updates and updates["yolo_conf"] is not None:
        val = float(updates["yolo_conf"])
        if not (0.05 <= val <= 0.95):
            raise ValueError("yolo_conf must be between 0.05 and 0.95.")
        cfg["yolo_conf"] = val

    path = get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def reset_settings() -> Dict[str, Any]:
    """Resets configuration back to lightweight CPU defaults."""
    return save_settings(dict(DEFAULT_SETTINGS))


def detect_hardware_profile() -> Dict[str, Any]:
    """Detects PyTorch CUDA vs CPU-only runtime accurately without paying a multi-second PyTorch import penalty."""
    import sys
    import shutil
    import importlib.util
    import re

    has_cuda = False
    gpu_name = None
    torch_build = "unknown"
    torch_cuda_build = False

    if "torch" in sys.modules:
        try:
            t_mod = sys.modules["torch"]
            torch_build = str(getattr(t_mod, "__version__", "unknown"))
            torch_cuda_build = bool(getattr(getattr(t_mod, "version", None), "cuda", None))
            has_cuda = bool(t_mod.cuda.is_available())
            if has_cuda:
                gpu_name = t_mod.cuda.get_device_name(0)
        except Exception:
            pass
    else:
        # Read ~1 KB torch/version.py (< 2 ms) so +cpu builds never report false CUDA availability
        try:
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                vpath = os.path.join(os.path.dirname(spec.origin), "version.py")
                if os.path.exists(vpath):
                    with open(vpath, "r", encoding="utf-8") as vf:
                        vtxt = vf.read()
                    m_ver = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", vtxt)
                    m_cuda = re.search(r"cuda[^=]*=\s*(None|['\"][^'\"]+['\"])", vtxt)
                    if m_ver:
                        torch_build = m_ver.group(1)
                    if m_cuda and m_cuda.group(1) != "None":
                        torch_cuda_build = True
        except Exception:
            pass

        if torch_cuda_build:
            try:
                import ctypes
                if os.name == "nt":
                    ctypes.WinDLL("nvcuda.dll")
                    if shutil.which("nvidia-smi"):
                        has_cuda = True
                        gpu_name = f"NVIDIA CUDA ({torch_build})"
                elif os.path.exists("/dev/nvidia0") or shutil.which("nvidia-smi"):
                    has_cuda = True
                    gpu_name = f"NVIDIA CUDA ({torch_build})"
            except Exception:
                has_cuda = False

    cpu_count = os.cpu_count() or 1
    return {
        "device": "cuda" if has_cuda else "cpu",
        "has_cuda": has_cuda,
        "torch_build": torch_build,
        "gpu_name": gpu_name,
        "cpu_cores": cpu_count,
    }


def get_hardware_tips(settings: Optional[Dict[str, Any]] = None) -> List[str]:
    """Generates practical, actionable tips tailored to the user's current settings & hardware."""
    cfg = settings or load_settings()
    hw = detect_hardware_profile()
    tips: List[str] = []

    scene_m = cfg.get("scene_model", "mobileclip2-s0")
    info = MODEL_CATALOG.get(scene_m, MODEL_CATALOG["mobileclip2-s0"])

    if not hw["has_cuda"]:
        if info.get("gpu_recommended"):
            tips.append(
                f"Warning: '{scene_m}' ({info['tier']}) requires high CPU/RAM ({info['ram_estimate']}); "
                "for faster CPU runs use 'mobileclip2-s0' or 'mobileclip2-s2'."
            )
        else:
            tips.append(
                f"CPU-Only environment ({hw['cpu_cores']} cores): '{scene_m}' ({info['tier']}) fits well ({info['cpu_speed']})."
            )
        if cfg.get("enable_smolvlm2"):
            tips.append(
                "Note: SmolVLM2 (GENERATIVE VLM / HEAVY) is ON in CPU mode (~10-35s per keyframe). "
                "Best for single images or short clips; turn it off with `vidstd config set --smolvlm2 off` for long videos."
            )
    else:
        tips.append(
            f"GPU detected ({hw['gpu_name']}): heavy variants like 'mobileclip2-l-14' or '--smolvlm2' will run quickly."
        )

    tips.append("Quick override per command without changing saved settings: pass `--model <name>` or `--smolvlm2`.")
    return tips


def get_model_catalog() -> Dict[str, Any]:
    """Returns full model catalog with live availability status, active flags, and hardware tips."""
    cfg = load_settings()
    hw = detect_hardware_profile()
    models_out: List[Dict[str, Any]] = []

    for m_id, info in MODEL_CATALOG.items():
        entry = dict(info)
        entry["active"] = (cfg["scene_model"] == m_id)
        entry["downloaded_locally"] = is_model_downloaded_locally(m_id)
        models_out.append(entry)

    vlm_entry = dict(SMOLVLM2_INFO)
    vlm_entry["active"] = bool(cfg.get("enable_smolvlm2", False))
    vlm_entry["downloaded_locally"] = is_model_downloaded_locally("smolvlm2")

    return {
        "settings_path": get_settings_path(),
        "current_settings": cfg,
        "hardware": hw,
        "scene_models": models_out,
        "generative_vlm": vlm_entry,
        "tips": get_hardware_tips(cfg),
    }
