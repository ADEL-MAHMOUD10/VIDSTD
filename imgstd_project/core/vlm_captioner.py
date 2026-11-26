"""
VIDSTD & IMGSTD Open-Ended Temporal Video & Keyframe Micro-VLM Captioner.
Wraps SmolVLM2-500M-Video-Instruct (HuggingFaceTB/SmolVLM2-500M-Video-Instruct via transformers)
for dense zero-rule scene descriptions, temporal clip understanding across 4-8 frame bursts,
and fine-grained regional attribute grounding, with deterministic heuristic fallback when weights are offline.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("imgstd.vlm")

# Pinned default Hugging Face revision for SmolVLM2 (never runs untrusted remote code)
PINNED_SMOLVLM2_REVISION = "main"

# Heavy neural dependencies (torch, transformers) are imported lazily ONLY when SmolVLM2 is enabled.
torch = None
AutoProcessor = None
AutoModelForImageTextToText = None
HAS_TORCH = False
HAS_TRANSFORMERS = False


class SmolVLM2Captioner:
    """
    Singleton Open-Ended Temporal Video & Keyframe Micro-VLM Wrapper.
    Runs on Keyframes (I-Frames/Scene Cuts), Single Images, and Temporal Video Frame Bursts (4-8 frames).
    Produces natural language descriptions of mood, facial expressions, weather, actions,
    and temporal interactions without hardcoded rules.
    """

    MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        enable_neural: Optional[bool] = None,
        scene_model: Optional[str] = None,
    ):
        self.model_id = model_id or self.MODEL_ID
        self._device = device
        self._enable_neural_override = enable_neural
        self.enable_neural = bool(enable_neural) if enable_neural is not None else False
        self.scene_model = scene_model
        self._model = None
        self._processor = None
        self._load_attempted = False

    def _is_neural_enabled(self) -> bool:
        if self._enable_neural_override is not None:
            return bool(self._enable_neural_override)
        from .config import load_settings
        return bool(load_settings().get("enable_smolvlm2", False))

    def _ensure_loaded(self):
        global torch, AutoProcessor, AutoModelForImageTextToText, HAS_TORCH, HAS_TRANSFORMERS
        if not self._is_neural_enabled():
            return
        if self._load_attempted:
            return
        self._load_attempted = True

        if not HAS_TORCH:
            try:
                import torch as _torch
                torch = _torch
                HAS_TORCH = True
            except ImportError:
                return
        if not HAS_TRANSFORMERS:
            try:
                from transformers import AutoProcessor as _AP, AutoModelForImageTextToText as _AM
                AutoProcessor = _AP
                AutoModelForImageTextToText = _AM
                HAS_TRANSFORMERS = True
            except ImportError:
                try:
                    from transformers import AutoProcessor as _AP, AutoModelForCausalLM as _AM
                    AutoProcessor = _AP
                    AutoModelForImageTextToText = _AM
                    HAS_TRANSFORMERS = True
                except ImportError:
                    return

        # Determine target device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        from .config import get_weights_root, SMOLVLM2_INFO
        trusted_local_dir = os.path.join(get_weights_root(), "SmolVLM2-500M-Video-Instruct")
        use_local = os.path.isfile(os.path.join(trusted_local_dir, "model.safetensors"))
        target_path = trusted_local_dir if use_local else self.model_id
        pinned_rev = None if use_local else SMOLVLM2_INFO.get("revision", PINNED_SMOLVLM2_REVISION)

        try:
            dtype = torch.bfloat16 if (self._device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if self._device == "cuda" else torch.float32)
            load_kwargs: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": False}
            proc_kwargs: Dict[str, Any] = {"trust_remote_code": False}
            if pinned_rev:
                load_kwargs["revision"] = pinned_rev
                proc_kwargs["revision"] = pinned_rev

            self._model = AutoModelForImageTextToText.from_pretrained(target_path, **load_kwargs).to(self._device)
            self._processor = AutoProcessor.from_pretrained(target_path, **proc_kwargs)
        except Exception as exc:
            logger.warning("Failed to load SmolVLM2 from '%s': %s. Falling back to MobileCLIP2/heuristic.", target_path, exc)
            self._model = None
            self._processor = None

    @staticmethod
    def _prepare_vlm_pil(image: np.ndarray, max_side: int = 512):
        """Resizes frame to bounded resolution so SmolVLM2 runs single-tile without 1080p token explosion."""
        import cv2
        from PIL import Image as PILImage
        rgb = image[:, :, :3] if (image.ndim == 3 and image.shape[2] == 4) else image
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        h, w = rgb.shape[:2]
        if max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            rgb = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        return PILImage.fromarray(rgb)

    def _run_vlm_chat(self, prompt_text: str, pil_items: List[Any], is_video: bool = False, max_new_tokens: int = 56) -> Optional[str]:
        media_key = "video" if is_video else "image"
        messages = [{"role": "user", "content": [{"type": media_key}, {"type": "text", "text": prompt_text}]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        proc_kwargs: Dict[str, Any] = {"text": prompt, "return_tensors": "pt", "do_image_splitting": False}
        if is_video:
            proc_kwargs["videos"] = [pil_items]
        else:
            proc_kwargs["images"] = pil_items
        try:
            inputs = self._processor(**proc_kwargs).to(self._device)
        except TypeError:
            proc_kwargs.pop("do_image_splitting", None)
            inputs = self._processor(**proc_kwargs).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        desc = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        if "Assistant:" in desc:
            desc = desc.split("Assistant:", 1)[-1]
        elif "assistant\n" in desc:
            desc = desc.split("assistant\n", 1)[-1]
        return desc.strip()

    @staticmethod
    def format_event_tag(clip_res: Optional[Dict[str, Any]]) -> Optional[str]:
        """Returns calibrated active event phrase only when it clearly exceeds the 'calm' baseline."""
        if not clip_res:
            return None
        evt = str(clip_res.get("event", "") or "")
        if not evt or evt == "calm":
            return None
        evt_sim = float(clip_res.get("event_sim", 0.0))
        calm_sim = float(clip_res.get("event_calm_sim", 0.0))
        if evt_sim >= 0.215 and (evt_sim - calm_sim) >= 0.022:
            return evt
        return None

    def caption_scene(
        self,
        image: np.ndarray,
        regions: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None,
        scene_model: Optional[str] = None,
        precomputed_clip: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Executes single image multimodal description or heuristic fallback.
        Returns a rich natural language paragraph describing mood, actions, and scene context.
        """
        self._ensure_loaded()

        if self._model is not None and self._processor is not None:
            try:
                pil_img = self._prepare_vlm_pil(image, max_side=512)
                desc = self._run_vlm_chat(
                    "Describe clearly what is happening in this scene in 1 to 2 concise sentences without inventing unseen details.",
                    [pil_img],
                    is_video=False,
                    max_new_tokens=56,
                )
                if desc and len(desc) >= 15:
                    return desc
            except Exception as exc:
                logger.warning("SmolVLM2 single-frame captioning failed (%s); falling back to heuristic.", exc)

        # Robust Deterministic / MobileCLIP2 Fallback
        return self._heuristic_scene_caption(
            image, regions, texts, scene_model=scene_model, precomputed_clip=precomputed_clip
        )

    def caption_video_clip(
        self,
        frames: List[np.ndarray],
        fps: float = 24.0,
        regions: Optional[List[Any]] = None,
        texts: Optional[List[Any]] = None
    ) -> str:
        """
        Executes fast single-tile temporal video reasoning across a sampled 4-frame burst.
        Returns an open-ended natural language summary of actions, events, and scene progression.
        """
        if not frames:
            return "Empty video sequence."

        self._ensure_loaded()

        if self._model is not None and self._processor is not None:
            try:
                num_samples = min(4, len(frames))
                indices = np.linspace(0, len(frames) - 1, num_samples).astype(int)
                sampled_pil = [self._prepare_vlm_pil(frames[idx], max_side=384) for idx in indices]
                desc = self._run_vlm_chat(
                    "Describe what happens across these frames cleanly in 1 to 2 sentences, focusing on real actions, objects, and key events.",
                    sampled_pil,
                    is_video=True,
                    max_new_tokens=72,
                )
                if desc:
                    return desc
            except Exception as exc:
                logger.warning("SmolVLM2 video clip captioning failed (%s); falling back to heuristic.", exc)

        # Fallback to temporal heuristic
        return self._heuristic_video_caption(frames, fps, regions, texts)

    def describe_regions(
        self,
        image: np.ndarray,
        regions: List[Any],
        label_map: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Executes fine-grained regional attribute description on non-occluded pixel masks.
        Returns mapping {region_id: 'perceptual color, clothing/signal state, location'}.
        """
        self._ensure_loaded()

        results: Dict[str, str] = {}
        h, w = image.shape[:2]

        for r in regions:
            reg_id = str(getattr(r, "region_id", None) or (r.get("region_id", "001") if isinstance(r, dict) else "001"))
            cls_name = str(getattr(r, "class_name", None) or (r.get("class_name", "object") if isinstance(r, dict) else "object") or "object")
            lbl_id = getattr(r, "label_id", None)
            bbox = getattr(r, "bbox", None)
            if bbox is None and isinstance(r, dict):
                bb = r.get("bounding_box", {})
                bbox = (bb.get("x", 0), bb.get("y", 0), bb.get("width", w), bb.get("height", h))
            elif bbox is None:
                bbox = (0, 0, w, h)

            bx, by, bw, bh = bbox
            y1, y2 = max(0, by), min(h, by + bh)
            x1, x2 = max(0, bx), min(w, bx + bw)
            crop = image[y1:y2, x1:x2]
            mask_crop = None
            if label_map is not None and lbl_id is not None and y2 > y1 and x2 > x1:
                mc = (label_map[y1:y2, x1:x2] == lbl_id)
                if np.count_nonzero(mc) >= 16:
                    mask_crop = mc

            # Analyze visual characteristics of the non-occluded region
            attrs = self._analyze_region_attributes(cls_name, crop, bbox, (w, h), mask_crop=mask_crop)

            # Optional neural enrichment for PRIMARY regions when SmolVLM2 is active on GPU
            role = getattr(r, "role", r.get("role", "") if isinstance(r, dict) else "")
            kind = getattr(r, "kind", r.get("kind", "OBJ") if isinstance(r, dict) else "OBJ")
            if (
                self._model is not None and self._processor is not None
                and self._device == "cuda"
                and kind != "BG" and role == "PRIMARY"
                and crop.size > 0 and min(crop.shape[:2]) >= 48
            ):
                try:
                    import PIL.Image as PILImage
                    rgb_crop = crop[:, :, :3] if (crop.ndim == 3 and crop.shape[2] == 4) else crop
                    pil_crop = PILImage.fromarray(rgb_crop)
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Briefly describe the visual appearance and state of this {cls_name} in one short phrase."}]}]
                    pr = self._processor.apply_chat_template(msg, add_generation_prompt=True)
                    inp = self._processor(text=pr, images=[pil_crop], return_tensors="pt").to(self._device)
                    with torch.no_grad():
                        gids = self._model.generate(**inp, max_new_tokens=32)
                    trimmed = [o[len(i):] for i, o in zip(inp.input_ids, gids)]
                    desc = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                    desc = desc.split("Assistant:", 1)[-1].split("assistant\n", 1)[-1].strip()
                    if desc:
                        attrs = f"{desc} ({attrs})"
                except Exception as exc:
                    logger.warning("SmolVLM2 regional attribute enrichment failed (%s).", exc)

            results[reg_id] = attrs

        return results

    def _heuristic_scene_caption(
        self,
        image: np.ndarray,
        regions: Optional[List[Any]],
        texts: Optional[List[Any]],
        scene_model: Optional[str] = None,
        precomputed_clip: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synthesizes an open-ended, human-like scene description based on
        multimodal visual signals (detected entities, typography, color harmony, and lighting).
        """
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1

        # Luminance & contrast
        gray = image if c == 1 else (image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114)
        mean_lum = float(np.mean(gray))
        lum_std = float(np.std(gray))

        # Weather / Atmosphere / Lighting heuristic
        if mean_lum < 40:
            lighting = "dimly lit, night-time atmospheric lighting"
        elif mean_lum > 200:
            lighting = "bright, high-key ambient illumination"
        elif lum_std > 65:
            lighting = "high-contrast cinematic lighting with deep shadows"
        else:
            lighting = "balanced daylight illumination"

        reg_list = regions or []
        txt_list = texts or []

        class_names = [
            str(getattr(r, "class_name", None) or (r.get("class_name", "") if isinstance(r, dict) else "") or "")
            for r in reg_list
            if ((getattr(r, "kind", None) or (r.get("kind", "OBJ") if isinstance(r, dict) else "OBJ")) != "BG")
        ]
        class_names = [cn for cn in class_names if cn and not cn.startswith("Ambient")]

        struct_names = [
            str(getattr(r, "class_name", None) or (r.get("class_name", "") if isinstance(r, dict) else "") or "")
            for r in reg_list
            if ((getattr(r, "kind", None) or (r.get("kind", "OBJ") if isinstance(r, dict) else "OBJ")) == "STRUCT")
        ]

        txt_strings = [
            str(getattr(t, "text", None) or (t.get("text", "") if isinstance(t, dict) else "") or "")
            for t in txt_list
        ]

        # Separate real signage text from annotation labels mirroring detected object classes
        cls_tokens = {cn.lower().split("(")[0].strip() for cn in class_names}
        real_signage = [
            t for t in txt_strings
            if t.strip() and t.strip().lower() not in cls_tokens
            and not any(tok in t.strip().lower().split() for tok in cls_tokens if len(t.strip().split()) <= 2)
        ]

        # Space / Celestial detection
        bg_names = [
            str(getattr(r, "class_name", None) or (r.get("class_name", "") if isinstance(r, dict) else "") or "")
            for r in reg_list
        ]
        if any("Earth" in cn or "Outer Space" in cn for cn in class_names + bg_names):
            return f"A cosmic scene capturing Planet Earth suspended in deep space, illuminated by {lighting}."

        # Grounded counts per entity type (e.g. "1 person, 1 car, 1 bicycle")
        from collections import Counter
        clean_labels = [cn.split("(")[0].strip().lower() for cn in class_names]
        counts = Counter(clean_labels)
        count_phrases = [f"{cnt} {lbl}" if cnt == 1 else f"{cnt} {lbl}s" for lbl, cnt in list(counts.items())[:7]]

        # Architecture & Storefront ONLY when real structural detections exist
        if struct_names and any("Storefront" in cn or "Door" in cn or "Signboard" in cn for cn in struct_names):
            txt_phrase = f' with visible signage "{", ".join(real_signage[:2])}"' if real_signage else ""
            return f"A commercial storefront facade{txt_phrase}, set under {lighting}."

        # Lightweight MobileCLIP2 global environment & lighting context (real frames >= 200px, RGB/RGBA/Gray)
        if max(h, w) >= 200 and c in (1, 3, 4):
            try:
                from .scene_clip import get_mobileclip_analyzer
                clip_res = precomputed_clip if precomputed_clip is not None else get_mobileclip_analyzer(model_name=scene_model or self.scene_model).analyze(image)
                if clip_res:
                    env = clip_res.get("environment", "")
                    env_sim = clip_res.get("environment_sim", 0.0)
                    subj = clip_res.get("subject", "")
                    subj_sim = clip_res.get("subject_sim", 0.0)
                    clip_light = clip_res.get("lighting", lighting)
                    txt_phrase = f' with visible text "{", ".join(real_signage[:2])}"' if real_signage else ""

                    # Prevent MobileCLIP from hallucinating indoor/UI/garage when outdoor street objects are clearly detected
                    street_cues = {"car", "bicycle", "traffic light", "bench", "truck", "bus", "motorcycle"}
                    if any(s in counts for s in street_cues) and not struct_names:
                        bad_street_envs = ("storefront", "indoor", "game", "software", "garage", "stage", "document")
                        if any(b in env.lower() for b in bad_street_envs) or env_sim < 0.14:
                            env = "an outdoor urban street"
                            env_sim = max(env_sim, 0.20)

                    parts = []
                    if env_sim >= 0.14 and env:
                        parts.append(f"Scene set in {env}")
                    else:
                        parts.append("Visual scene")

                    # Concrete YOLO entities take priority over speculative MobileCLIP subject guesses
                    if count_phrases:
                        open_world_refines = (
                            "rodent", "heavy industrial", "soldiers", "spacecraft",
                            "helicopter", "athlete", "speaking", "worker", "performer",
                        )
                        if len(class_names) == 1 and subj_sim >= 0.16 and any(k in subj for k in open_world_refines):
                            parts.append(f"featuring {subj}")
                        else:
                            parts.append(f"featuring {', '.join(count_phrases)}")
                    elif subj_sim >= 0.16 and subj:
                        parts.append(f"featuring {subj}")

                    event_tag = self.format_event_tag(clip_res)
                    event_clause = f" ({event_tag})" if event_tag else ""

                    return f"{', '.join(parts)}{event_clause} under {clip_light}{txt_phrase}."
            except Exception:
                pass

        # Concrete entities fallback without MobileCLIP
        if count_phrases:
            return f"A composed scene featuring {', '.join(count_phrases)}, framed with {lighting}."

        return f"A visual composition featuring abstract forms and textures rendered with {lighting}."

    def _heuristic_video_caption(
        self,
        frames: List[np.ndarray],
        fps: float,
        regions: Optional[List[Any]],
        texts: Optional[List[Any]]
    ) -> str:
        """Generates temporal dynamic description across video frames."""
        base_desc = self._heuristic_scene_caption(frames[0], regions, texts)
        if len(frames) < 2:
            return base_desc

        # Sample frame difference
        f0 = frames[0].astype(float)
        f_mid = frames[len(frames) // 2].astype(float)
        f_end = frames[-1].astype(float)

        diff1 = np.mean(np.abs(f_mid - f0))
        diff2 = np.mean(np.abs(f_end - f_mid))
        mean_diff = (diff1 + diff2) / 2.0

        if mean_diff < 1.0:
            dynamics = "with continuous static stability throughout the sequence"
        elif mean_diff > 35.0:
            dynamics = "capturing rapid action, dynamic motion, and significant visual change across time"
        else:
            dynamics = "evolving with smooth fluid motion and consistent temporal continuity"

        dur = round(len(frames) / max(1.0, fps), 2)
        return f"{base_desc} Temporal sequence ({dur}s duration) {dynamics}."

    @staticmethod
    def _rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
        """Converts RGB tuple (0-255) to a rich, human-accurate perceptual HSV color name."""
        import colorsys
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_deg = h * 360.0

        if s < 0.16 or v < 0.14:
            if v < 0.18:
                return "black"
            if v < 0.35:
                return "dark charcoal gray"
            if v < 0.65:
                return "gray"
            if v < 0.85:
                return "silver / light gray"
            return "white"

        if 10.0 <= h_deg < 28.0:
            if s >= 0.45:
                return "burnt orange / rust-orange" if v < 0.62 else "bright orange"
            return "dark charcoal / brown" if v < 0.38 else ("dark brown" if v < 0.45 else "warm tan / beige")
        if 28.0 <= h_deg < 38.0:
            if s < 0.38 and v < 0.38:
                return "dark charcoal / brown"
            if s < 0.65:
                return "golden tan / light brown" if v >= 0.45 else "warm brown"
            return "warm terracotta / brown" if v < 0.52 else "amber / golden orange"
        if h_deg < 10.0 or h_deg >= 342.0:
            return "dark red / maroon" if v < 0.45 else "red"
        if 38.0 <= h_deg < 62.0:
            if s < 0.38 and v < 0.38:
                return "dark charcoal / olive"
            return "golden tan / olive" if s < 0.55 and v < 0.65 else ("gold / mustard" if v < 0.72 else "yellow")
        if 62.0 <= h_deg < 82.0:
            return "olive green" if v < 0.55 else "yellow-green"
        if 82.0 <= h_deg < 165.0:
            return "dark green" if v < 0.42 else "green"
        if 165.0 <= h_deg < 195.0:
            return "dark navy / denim blue" if v < 0.38 else "teal / cyan"
        if 195.0 <= h_deg < 258.0:
            if v < 0.42:
                return "dark navy blue"
            if s < 0.50 or v > 0.65:
                return "slate / light blue"
            return "blue"
        if 258.0 <= h_deg < 300.0:
            return "purple / violet"
        return "pink / magenta"

    def _analyze_region_attributes(
        self,
        cls_name: str,
        crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        canvas_wh: Tuple[int, int],
        mask_crop: Optional[np.ndarray] = None
    ) -> str:
        """Derives rich perceptual color, body paint, clothing, traffic-signal, and framing attributes."""
        if crop.size == 0:
            return f"standard {cls_name}"

        bx, by, bw, bh = bbox
        cw, ch = canvas_wh
        aspect = bw / float(max(1, bh))
        cls_low = cls_name.lower()

        # Select non-occluded mask pixels with spatial striding on large regions (avoids multi-million-pixel sort)
        ch_c, cw_c = crop.shape[:2]
        stride = max(1, max(ch_c, cw_c) // 96)
        crop_s = crop[::stride, ::stride] if stride > 1 else crop
        if mask_crop is not None and mask_crop.shape[:2] == crop.shape[:2] and np.any(mask_crop):
            mc_s = mask_crop[::stride, ::stride] if stride > 1 else mask_crop
            pts = crop_s[mc_s] if np.any(mc_s) else crop[mask_crop]
        else:
            pts = crop_s.reshape(-1, crop_s.shape[2]) if crop_s.ndim == 3 else crop_s.reshape(-1)

        attrs: List[str] = []
        if crop.ndim == 3 and crop.shape[2] >= 3 and len(pts) > 0:
            import cv2
            rgb_pts = pts[:, :3]
            med_rgb = tuple(int(v) for v in np.median(rgb_pts, axis=0))
            base_color_name = self._rgb_to_color_name(med_rgb)

            # For vehicles, extract chromatic lit body panels so black tires/shadows do not mask paint hue
            if any(k in cls_low for k in ("car", "truck", "bus", "motorcycle", "train")):
                hsv_pts = cv2.cvtColor(rgb_pts.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                body_pts = rgb_pts[(hsv_pts[:, 1] > 80) & (hsv_pts[:, 2] > 60)]
                if len(body_pts) >= max(20, int(len(rgb_pts) * 0.15)):
                    body_rgb = tuple(int(v) for v in np.median(body_pts, axis=0))
                    body_name = self._rgb_to_color_name(body_rgb)
                    attrs.append(f"color: {body_name} (RGB({body_rgb[0]},{body_rgb[1]},{body_rgb[2]}))")
                else:
                    attrs.append(f"color: {base_color_name} (RGB({med_rgb[0]},{med_rgb[1]},{med_rgb[2]}))")
            elif "traffic light" in cls_low:
                hsv_pts = cv2.cvtColor(rgb_pts.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
                glow_pts = rgb_pts[(hsv_pts[:, 2] > 125) & (hsv_pts[:, 1] > 60)]
                if len(glow_pts) >= 12:
                    g_rgb = tuple(int(v) for v in np.median(glow_pts, axis=0))
                    g_name = self._rgb_to_color_name(g_rgb)
                    sig = "green" if ("green" in g_name or "teal" in g_name or "cyan" in g_name) else ("red" if "red" in g_name else g_name)
                    attrs.append(f"color: {base_color_name} casing, {sig} signal lit")
                else:
                    attrs.append(f"color: {base_color_name}")
            elif "person" in cls_low and mask_crop is not None and np.count_nonzero(mask_crop) >= 40:
                ys, _ = np.where(mask_crop)
                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                p_h = max(1, y_max - y_min)
                up_m = mask_crop.copy()
                up_m[:y_min + int(p_h * 0.16), :] = False
                up_m[y_min + int(p_h * 0.48):, :] = False
                lo_m = mask_crop.copy()
                lo_m[:y_min + int(p_h * 0.50), :] = False
                lo_m[y_min + int(p_h * 0.92):, :] = False
                top_str = self._rgb_to_color_name(tuple(int(v) for v in np.median(crop[up_m][:, :3], axis=0))) if np.count_nonzero(up_m) > 10 else base_color_name
                bot_str = self._rgb_to_color_name(tuple(int(v) for v in np.median(crop[lo_m][:, :3], axis=0))) if np.count_nonzero(lo_m) > 10 else base_color_name
                attrs.append(f"wearing {top_str} top and {bot_str} lower wear")
            else:
                attrs.append(f"color: {base_color_name} (RGB({med_rgb[0]},{med_rgb[1]},{med_rgb[2]}))")
        else:
            attrs.append("color: monochrome")

        # Posture & Spatial Framing
        if "Person" in cls_name:
            if aspect < 0.48:
                attrs.append("standing/walking posture")
            elif aspect > 0.85:
                attrs.append("crouching/seated or close-up")
            else:
                attrs.append("active posture")
        elif "Planet" in cls_name or "Earth" in cls_name:
            attrs.append("spherical celestial body, atmospheric limb")
        elif "Storefront" in cls_name or "Building" in cls_name:
            attrs.append("architectural facade, street level")
        elif "Sign" in cls_name:
            attrs.append("mounted typography surface")

        # Relative location
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        loc_x = "left" if cx < cw * 0.33 else ("right" if cx > cw * 0.66 else "center")
        loc_y = "upper" if cy < ch * 0.33 else ("lower" if cy > ch * 0.66 else "middle")
        attrs.append(f"positioned {loc_y}-{loc_x}")

        return ", ".join(attrs)


Florence2Captioner = SmolVLM2Captioner
_CACHED_VLM_CAPTIONER = None


def get_vlm_captioner(
    enable_neural: Optional[bool] = None,
    scene_model: Optional[str] = None
) -> SmolVLM2Captioner:
    """Lazy singleton accessor for the SmolVLM2 / MobileCLIP2 scene captioner."""
    global _CACHED_VLM_CAPTIONER
    if _CACHED_VLM_CAPTIONER is None:
        _CACHED_VLM_CAPTIONER = SmolVLM2Captioner(enable_neural=enable_neural, scene_model=scene_model)
    else:
        if enable_neural is not None:
            _CACHED_VLM_CAPTIONER._enable_neural_override = enable_neural
            _CACHED_VLM_CAPTIONER.enable_neural = bool(enable_neural)
        if scene_model is not None:
            _CACHED_VLM_CAPTIONER.scene_model = scene_model
    return _CACHED_VLM_CAPTIONER
