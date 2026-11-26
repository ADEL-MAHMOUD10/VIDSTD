"""
IMGSTD Lightweight Scene Context Classifier — MobileCLIP2-S0 (Visual-Only Runtime).
Runs FastViT-MCI0 (~44 MB visual weights, ~93 MB RAM total, ~75 ms on CPU)
with precomputed multi-axis text embeddings (~100 KB) to provide grounded
global environment, subject, and lighting understanding without VLM token generation.
"""

import os
import gc
import json
import time
import hashlib
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("imgstd.scene_clip")

torch = None
HAS_TORCH = False


def _ensure_torch_imported() -> bool:
    global torch, HAS_TORCH
    if HAS_TORCH and torch is not None:
        return True
    try:
        import torch as _torch
        torch = _torch
        HAS_TORCH = True
        return True
    except ImportError:
        return False


SCENE_PROMPT_AXES: Dict[str, List[Tuple[str, str]]] = {
    "environment": [
        ("sandy desert landscape with dunes", "a photo of a sandy desert landscape with dunes"),
        ("open ocean or sea", "a photo of an ocean or sea with water waves"),
        ("outer space with planet earth", "a photo of outer space with planet earth and stars"),
        ("dense green forest", "a photo of a dense green forest with trees"),
        ("snowy winter landscape", "a photo of snowy mountains and winter landscape"),
        ("rocky mountain or canyon", "a photo of rocky mountains and cliffs"),
        ("city street with buildings", "a photo of a city street with buildings"),
        ("modern downtown skyline", "a photo of tall city skyscrapers and urban skyline"),
        ("commercial storefront facade", "a photo of a commercial storefront with signage"),
        ("indoor room or office", "a photo of an indoor room or office"),
        ("living room or home interior", "a photo of a furnished home living room or bedroom"),
        ("kitchen or dining room", "a photo of a home kitchen or restaurant dining area"),
        ("industrial facility or warehouse", "a photo of an industrial factory or warehouse"),
        ("paved road or highway", "a photo of a paved road or highway"),
        ("train station or railway", "a photo of a railway track or train station"),
        ("airport or airfield", "a photo of an airport runway or hangar"),
        ("grassy field or park", "a photo of a green grassy field or park"),
        ("beach and coastline", "a photo of a sandy beach meeting the sea"),
        ("sports stadium or athletic court", "a photo of a sports stadium, gym, or playing field"),
        ("underwater marine scene", "an underwater ocean scene"),
        ("laboratory or medical room", "a photo of a science laboratory, clinic, or hospital room"),
        ("dark cave or underground tunnel", "a photo of a dark cave or underground tunnel"),
        ("UI screen, chart, or document", "a digital user interface screen, chart, or text document"),
        ("minimal studio backdrop", "a clean studio photo backdrop or plain wall"),
    ],
    "subject": [
        ("small desert rodent or mouse", "a photo of a small mouse or desert rodent"),
        ("dog", "a photo of a dog"),
        ("cat", "a photo of a cat"),
        ("bird", "a photo of a bird flying or perched"),
        ("horse or large wildlife", "a photo of a horse, deer, or wild mammal"),
        ("people or crowd", "a photo of people standing or walking"),
        ("people talking or meeting", "a photo of people conversing or working together"),
        ("armored figures or soldiers", "a photo of soldiers or armored people crouching or standing"),
        ("single person portrait", "a close-up portrait of one person"),
        ("athlete running or playing sports", "a photo of an athlete running, jumping, or playing sport"),
        ("giant heavy industrial machine or vehicle", "a photo of a massive heavy industrial machine or giant desert vehicle"),
        ("car or truck", "a photo of a passenger car or truck"),
        ("motorcycle or bicycle", "a photo of a bicycle or motorcycle"),
        ("flying aircraft or helicopter", "a photo of a small aircraft or helicopter in flight"),
        ("futuristic spacecraft", "a photo of a futuristic sci-fi spacecraft"),
        ("train on tracks", "a photo of a train on railway tracks"),
        ("boat or ship on water", "a photo of a boat or ship on water"),
        ("food dish or plated meal", "a photo of a prepared meal, dish, or dessert on a table"),
        ("smartphone, laptop, or electronics", "a photo of a laptop, phone, monitor, or electronic device"),
        ("book, sign, or printed page", "a photo of a printed book, signboard, or document"),
        ("architectural building or door", "a photo of a building exterior, door, or architectural facade"),
        ("metallic pillar or stake", "a photo of a vertical metallic stake or pillar"),
        ("furniture like chair, sofa, or table", "a photo of indoor furniture such as a chair, table, or couch"),
        ("abstract pattern or texture", "an abstract graphic pattern or solid color background"),
    ],
    "lighting": [
        ("bright sunlight with crisp shadows", "bright harsh sunlight with crisp shadows"),
        ("balanced daylight", "soft balanced natural daylight"),
        ("dimly lit dark shadows", "dimly lit night time scene with deep dark shadows"),
        ("warm sunset or golden hour glow", "warm golden sunset or sunrise lighting"),
        ("dusty or smoky hazy atmosphere", "dusty hazy atmosphere with smoke or sand in the air"),
        ("indoor artificial lighting", "indoor artificial or neon lighting"),
        ("high-contrast dramatic spotlight", "dramatic high-contrast studio or spotlight illumination"),
        ("overcast diffused lighting", "cloudy overcast cool diffused daylight"),
    ],
}


def _prompt_bank_signature() -> str:
    h = hashlib.sha256()
    for ax in sorted(SCENE_PROMPT_AXES.keys()):
        for label, text in SCENE_PROMPT_AXES[ax]:
            h.update(f"{ax}:{label}:{text}\n".encode("utf-8"))
    return h.hexdigest()[:16]


class MobileCLIPSceneAnalyzer:
    """
    Singleton low-RAM MobileCLIP2 visual encoder supporting S0/S2/S3/S4/B/L-14 variants.
    Strips the text transformer after caching prompt embeddings to a tiny ~100 KB .npz file.
    """

    def __init__(self, model_name: Optional[str] = None, weights_dir: Optional[str] = None):
        from .config import MODEL_CATALOG, normalize_scene_model_name
        self.model_id = normalize_scene_model_name(model_name)
        meta = MODEL_CATALOG[self.model_id]
        self.open_clip_name = meta["open_clip_name"]
        self.pretrained_tag = meta.get("pretrained_tag", "dfndr2b")
        self.weights_dir = weights_dir
        self._visual = None
        self._ort_session = None
        self._ort_input_name = None
        self._image_size = 256
        self._image_mean = (0.0, 0.0, 0.0)
        self._image_std = (1.0, 1.0, 1.0)
        self._prompt_bank: Optional[Dict[str, Tuple[List[str], np.ndarray]]] = None
        self._load_attempted = False

    def _find_weights_dir(self) -> Optional[str]:
        from .config import get_weights_root
        candidates = []
        if self.weights_dir:
            candidates.append(self.weights_dir)
        candidates.extend([
            os.path.join(os.getcwd(), "weights", self.open_clip_name),
            os.path.join(os.path.dirname(__file__), "..", "..", "weights", self.open_clip_name),
            os.path.join(get_weights_root(), self.open_clip_name),
        ])
        for cand in candidates:
            model_file = os.path.join(cand, "open_clip_model.safetensors")
            if os.path.exists(model_file):
                return os.path.abspath(cand)
        return None

    def _get_cache_dir(self) -> str:
        from .config import get_weights_root
        wdir = self._find_weights_dir()
        if wdir:
            return wdir
        w_root = os.path.join(get_weights_root(), self.open_clip_name)
        try:
            os.makedirs(w_root, exist_ok=True)
            return w_root
        except Exception:
            fallback = os.path.join(os.path.expanduser("~"), ".imgstd", "cache", self.open_clip_name)
            os.makedirs(fallback, exist_ok=True)
            return fallback

    def _load_prompt_bank_only(self, cache_path: str) -> bool:
        sig = f"{self.open_clip_name}:{_prompt_bank_signature()}"
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=False)
                if str(data["signature"].item()) == sig:
                    bank: Dict[str, Tuple[List[str], np.ndarray]] = {}
                    for ax in SCENE_PROMPT_AXES:
                        labels = [label for label, _ in SCENE_PROMPT_AXES[ax]]
                        bank[ax] = (labels, data[f"emb_{ax}"].astype(np.float32))
                    self._prompt_bank = bank
                    return True
            except Exception:
                pass
        return False

    def _build_prompt_bank_from_model(self, model: Any, cache_path: str) -> bool:
        sig = f"{self.open_clip_name}:{_prompt_bank_signature()}"
        try:
            import open_clip
            tokenizer = open_clip.get_tokenizer(self.open_clip_name)
            save_dict: Dict[str, Any] = {"signature": np.array(sig)}
            bank: Dict[str, Tuple[List[str], np.ndarray]] = {}
            with torch.no_grad():
                for ax, items in SCENE_PROMPT_AXES.items():
                    labels = [lbl for lbl, _ in items]
                    prompts = [txt for _, txt in items]
                    tf = model.encode_text(tokenizer(prompts))
                    tf = tf / tf.norm(dim=-1, keepdim=True)
                    arr = tf.cpu().numpy().astype(np.float32)
                    save_dict[f"emb_{ax}"] = arr
                    bank[ax] = (labels, arr)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, **save_dict)
            self._prompt_bank = bank
            return True
        except Exception:
            return False

    def _save_visual_meta(self, meta_path: str) -> None:
        tmp_path = f"{meta_path}.tmp.{os.getpid()}"
        meta = {
            "open_clip_name": self.open_clip_name,
            "image_size": int(self._image_size),
            "image_mean": list(self._image_mean),
            "image_std": list(self._image_std),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp_path, meta_path)

    def _load_visual_meta(self, meta_path: str) -> bool:
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("open_clip_name") != self.open_clip_name:
                return False
            self._image_size = int(meta["image_size"])
            self._image_mean = tuple(float(x) for x in meta["image_mean"][:3])
            self._image_std = tuple(float(x) for x in meta["image_std"][:3])
            return True
        except Exception:
            return False

    def _try_load_cached_visual(self, cache_dir: str) -> bool:
        meta_path = os.path.join(cache_dir, "visual_meta.json")
        if not self._load_visual_meta(meta_path):
            return False

        # 1. Prefer ONNXRuntime visual cache if present (zero PyTorch import overhead)
        onnx_path = os.path.join(cache_dir, "visual_model.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                self._ort_session = sess
                self._ort_input_name = sess.get_inputs()[0].name
                return True
            except Exception as exc:
                logger.debug("Cached visual ONNX load skipped (%s).", exc)
                self._ort_session = None

        # 2. Zero-open_clip TorchScript visual cache (~720ms vs ~8.4s open_clip/TimmModel cold init)
        ts_path = os.path.join(cache_dir, "visual_model.ts")
        if os.path.exists(ts_path) and _ensure_torch_imported():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vis_ts = torch.jit.load(ts_path, map_location="cpu")
                vis_ts.eval()
                self._visual = vis_ts
                return True
            except Exception as exc:
                logger.debug("Cached visual TorchScript load skipped (%s).", exc)
                self._visual = None

        return False

    def _export_visual_caches(self, cache_dir: str, vis: Any) -> None:
        """Persists lightweight visual-only TorchScript (and ONNX when available) caches."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            meta_path = os.path.join(cache_dir, "visual_meta.json")
            self._save_visual_meta(meta_path)

            dummy = torch.zeros(1, 3, self._image_size, self._image_size, dtype=torch.float32)
            ts_path = os.path.join(cache_dir, "visual_model.ts")
            if not os.path.exists(ts_path):
                tmp_ts = f"{ts_path}.tmp.{os.getpid()}"
                try:
                    with warnings.catch_warnings(), torch.no_grad():
                        warnings.simplefilter("ignore")
                        traced = torch.jit.trace(vis.eval(), dummy)
                        torch.jit.save(traced, tmp_ts)
                    os.replace(tmp_ts, ts_path)
                except Exception as exc:
                    if os.path.exists(tmp_ts):
                        try:
                            os.remove(tmp_ts)
                        except OSError:
                            pass
                    logger.debug("Could not export TorchScript visual cache (%s).", exc)

            onnx_path = os.path.join(cache_dir, "visual_model.onnx")
            if not os.path.exists(onnx_path):
                try:
                    import onnx  # noqa: F401
                    tmp_onnx = f"{onnx_path}.tmp.{os.getpid()}"
                    with warnings.catch_warnings(), torch.no_grad():
                        warnings.simplefilter("ignore")
                        torch.onnx.export(
                            vis.eval(),
                            dummy,
                            tmp_onnx,
                            input_names=["image"],
                            output_names=["embedding"],
                            opset_version=17,
                            do_constant_folding=True,
                            dynamo=False,
                        )
                    os.replace(tmp_onnx, onnx_path)
                except Exception:
                    pass
        except Exception:
            pass

    def _ensure_loaded(self) -> bool:
        if self._load_attempted:
            return (self._ort_session is not None or self._visual is not None) and self._prompt_bank is not None
        self._load_attempted = True

        wdir = self._find_weights_dir()
        cache_dir = self._get_cache_dir()
        cache_path = os.path.join(cache_dir, "prompt_bank_v1.npz")
        bank_ready = self._load_prompt_bank_only(cache_path)

        # Ultra-fast visual cache hit (skips open_clip/timm/safetensors completely)
        if bank_ready and self._try_load_cached_visual(cache_dir):
            return True

        if not _ensure_torch_imported():
            return False

        # Ultra-low-RAM fast path for local MobileCLIP2-S0 safetensors
        if self.open_clip_name == "MobileCLIP2-S0" and wdir is not None:
            model_path = os.path.join(wdir, "open_clip_model.safetensors")
            if not bank_ready:
                try:
                    import open_clip
                    tmp_m, _, _ = open_clip.create_model_and_transforms(
                        self.open_clip_name,
                        pretrained=model_path,
                        image_mean=(0.0, 0.0, 0.0),
                        image_std=(1.0, 1.0, 1.0),
                    )
                    tmp_m.eval()
                    bank_ready = self._build_prompt_bank_from_model(tmp_m, cache_path)
                    del tmp_m
                    gc.collect()
                except Exception:
                    return False

            if bank_ready:
                try:
                    from safetensors import safe_open
                    from open_clip.timm_model import TimmModel

                    vis = TimmModel(
                        "fastvit_mci0",
                        embed_dim=512,
                        image_size=(256, 256),
                        pool="avg",
                        proj=None,
                        pretrained=False,
                    )
                    with safe_open(model_path, framework="pt") as f:
                        vsd = {k[7:]: f.get_tensor(k) for k in f.keys() if k.startswith("visual.")}
                    vis.load_state_dict(vsd, strict=True)
                    vis.eval()
                    self._visual = vis
                    self._image_size = 256
                    self._export_visual_caches(cache_dir, vis)
                    return True
                except Exception:
                    self._visual = None

        # Generic path for MobileCLIP2-S2, S3, S4, B, L-14 (or downloaded weights)
        try:
            import open_clip
            cfg = open_clip.get_model_config(self.open_clip_name) or {}
            vcfg = cfg.get("vision_cfg", {})
            self._image_size = int(vcfg.get("image_size", 256) if not isinstance(vcfg.get("image_size"), (list, tuple)) else vcfg["image_size"][0])
            pretrained_arg = os.path.join(wdir, "open_clip_model.safetensors") if wdir is not None else self.pretrained_tag
            model, _, _ = open_clip.create_model_and_transforms(
                self.open_clip_name,
                pretrained=pretrained_arg,
            )
            model.eval()
            if not bank_ready:
                bank_ready = self._build_prompt_bank_from_model(model, cache_path)

            vis = model.visual
            if hasattr(vis, "image_mean") and getattr(vis, "image_mean", None):
                self._image_mean = tuple(float(x) for x in vis.image_mean[:3])
            if hasattr(vis, "image_std") and getattr(vis, "image_std", None):
                self._image_std = tuple(float(x) for x in vis.image_std[:3])

            # Strip heavy text encoder from RAM while keeping visual backbone intact
            for attr in ("transformer", "token_embedding", "positional_embedding", "ln_final", "text_projection", "text"):
                if hasattr(model, attr):
                    try:
                        delattr(model, attr)
                    except Exception:
                        pass
            del model
            gc.collect()

            if bank_ready:
                vis.eval()
                self._visual = vis
                self._export_visual_caches(cache_dir, vis)
                return True
        except Exception as exc:
            logger.warning("Failed to load MobileCLIP2 variant '%s' (%s); falling back to geometric heuristics.", self.open_clip_name, exc)
            self._visual = None

        return False

    def _preprocess_full_width_np(self, image: np.ndarray) -> np.ndarray:
        """Strips cinema letterbox black bars and pads full-width to model input resolution."""
        arr = image
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]

        h, w = arr.shape[:2]
        # Strip pure black letterbox top/bottom bars while keeping >= 40% height
        if h >= 64:
            row_means = arr.mean(axis=(1, 2))
            non_black = np.where(row_means >= 12.0)[0]
            if len(non_black) > int(h * 0.40):
                y1, y2 = int(non_black[0]), int(non_black[-1]) + 1
                if y2 - y1 >= int(h * 0.40):
                    arr = arr[y1:y2, :]

        sz = self._image_size
        pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        padded = ImageOps.pad(pil, (sz, sz), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        ten = np.asarray(padded, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.array(self._image_mean, dtype=np.float32)[:, None, None]
        std = np.array(self._image_std, dtype=np.float32)[:, None, None]
        ten = (ten - mean) / std
        return np.ascontiguousarray(ten[None, ...], dtype=np.float32)

    def _preprocess_full_width(self, image: np.ndarray) -> "torch.Tensor":
        ten_np = self._preprocess_full_width_np(image)
        if not _ensure_torch_imported():
            raise RuntimeError("PyTorch is required to build tensor input when ONNX cache is unavailable.")
        return torch.from_numpy(ten_np)

    def analyze(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Runs single-pass MobileCLIP2 visual encoding and returns top environment,
        subject, and lighting tags with similarity scores.
        """
        if not self._ensure_loaded():
            return None

        t0 = time.perf_counter()
        inp_np = self._preprocess_full_width_np(image)
        if self._ort_session is not None:
            raw_vec = self._ort_session.run(None, {self._ort_input_name: inp_np})[0][0].astype(np.float32)
            norm = float(np.linalg.norm(raw_vec))
            img_vec = raw_vec / max(1e-12, norm)
        else:
            with torch.no_grad():
                feat = self._visual(torch.from_numpy(inp_np))
                feat = feat / feat.norm(dim=-1, keepdim=True)
                img_vec = feat[0].cpu().numpy().astype(np.float32)

        out: Dict[str, Any] = {"model": self.model_id}
        for ax, (labels, emb_mat) in self._prompt_bank.items():
            sims = emb_mat @ img_vec
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])
            exp_s = np.exp((sims - float(np.max(sims))) * 100.0)
            probs = exp_s / float(np.sum(exp_s))
            out[ax] = labels[best_i]
            out[f"{ax}_sim"] = round(best_sim, 3)
            out[f"{ax}_prob"] = round(float(probs[best_i]), 3)

        out["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return out


_ANALYZER_CACHE: Dict[str, MobileCLIPSceneAnalyzer] = {}


def get_mobileclip_analyzer(model_name: Optional[str] = None) -> MobileCLIPSceneAnalyzer:
    """Lazy singleton accessor per MobileCLIP2 variant (defaults to saved settings['scene_model'])."""
    from .config import load_settings, normalize_scene_model_name
    target = normalize_scene_model_name(model_name or load_settings().get("scene_model", "mobileclip2-s0"))
    if target not in _ANALYZER_CACHE:
        _ANALYZER_CACHE[target] = MobileCLIPSceneAnalyzer(model_name=target)
    return _ANALYZER_CACHE[target]
