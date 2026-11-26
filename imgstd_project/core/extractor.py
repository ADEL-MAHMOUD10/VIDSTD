"""
IMGSTD Semantic Extractor & Label Map Module — V04.
Integrates True Semantic Instance Segmentation (YOLO11n-seg),
Open-Vocabulary Architectural & Structural Scene Detection,
Fast Multilingual OCR ([TXT] Layer), and Honest Background Modeling.
Prevents object hallucination and provides spatial containment/reflection reasoning.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

from .ocr import FastSceneOCR, TextDescriptor
from .embedding import RegionEmbeddingGenerator
from .vocab_maps import COCO_CLASS_ID_MAP, COCO_CLASS_NAMES, OPEN_VOCAB_MAP

logger = logging.getLogger("imgstd.extractor")


@dataclass
class RegionDescriptor:
    region_id: str
    class_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    dominant_color: List[int]        # [R, G, B] or [R, G, B, A]
    label_id: int                    # index into color LUT (1-indexed)
    area: int                        # Area in pixels
    coverage_pct: float              # Percentage of total canvas area
    kind: str = "OBJ"                # "OBJ" (physical entity), "STRUCT" (architecture), or "BG" (background)
    conf: float = 1.00               # Detection confidence (0.00 - 1.00)
    prominence: str = "SECONDARY"    # "PRIMARY", "SECONDARY", "MINOR_DETAIL"
    role: str = "SECONDARY"          # "PRIMARY", "SECONDARY", "MINOR_DETAIL"
    is_reflection: bool = False
    likely_reflection_or_inset: bool = False
    parent_structure: Optional[str] = None
    contained_in: Optional[str] = None
    class_name: Optional[str] = None
    category: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    attributes: Optional[str] = None


# COCO_CLASS_ID_MAP imported from .vocab_maps


class RLECodec:
    """Vectorized Run-Length Encoding / Decoding for binary masks (kept for V01 legacy)."""

    @staticmethod
    def encode(mask: np.ndarray) -> str:
        flat = mask.ravel().astype(np.uint8)
        n = len(flat)
        if n == 0:
            return ""
        diff_idx = np.where(flat[1:] != flat[:-1])[0] + 1
        starts = np.r_[0, diff_idx]
        lengths = np.diff(np.r_[starts, n])
        values = flat[starts]
        return ",".join(f"{int(v)}:{int(l)}" for v, l in zip(values, lengths))

    @staticmethod
    def decode(rle_str: str, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        expected_size = h * w
        if not rle_str or expected_size == 0:
            return np.zeros(shape, dtype=bool)
        parts = rle_str.split(",")
        runs = [p.split(":") for p in parts]
        vals = np.array([int(p[0]) for p in runs], dtype=bool)
        lens = np.array([int(p[1]) for p in runs], dtype=np.int32)
        flat = np.repeat(vals, lens)
        if len(flat) != expected_size:
            if len(flat) < expected_size:
                flat = np.pad(flat, (0, expected_size - len(flat)))
            else:
                flat = flat[:expected_size]
        return flat.reshape(shape)


class BaseSemanticExtractor(ABC):
    """Abstract interface for semantic region extractors."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> Tuple[List[RegionDescriptor], np.ndarray, np.ndarray]:
        """
        Extracts semantic regions from an image matrix (H, W, C).
        Returns:
            regions: List of RegionDescriptor
            label_map: H×W uint8 array where each pixel = label_id (0=background, 1-indexed)
            color_lut: (256, C) uint8 LUT mapping label_id -> dominant color
        """
        pass


# Global singleton cache for YOLO model
_CACHED_YOLO_MODEL = None


class OpenCVDNNYoloSeg:
    """
    Ultra-fast ONNX runner for YOLO11-seg models (prefers ORT CPUExecutionProvider,
    falls back to OpenCV DNN). Operates in ~80-110ms on CPU with zero PyTorch /
    Ultralytics dependencies, using aspect-ratio preserving letterboxing.
    """
    def __init__(self, onnx_path: str = "yolo11n-seg.onnx"):
        self._ort_session = None
        self._ort_input_name = None
        self.net = None
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._ort_session = sess
            self._ort_input_name = sess.get_inputs()[0].name
        except Exception:
            self.net = cv2.dnn.readNetFromONNX(onnx_path)

    def predict(
        self,
        image: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5
    ) -> List[Dict[str, Any]]:
        h_orig, w_orig = image.shape[:2]

        # 1. Letterbox preserving aspect ratio with constant padding (YOLO standard)
        scale = min(imgsz / w_orig, imgsz / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        dx = (imgsz - nw) // 2
        dy = (imgsz - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = cv2.resize(image, (nw, nh))

        if self._ort_session is not None:
            blob_ort = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
            blob_ort = np.ascontiguousarray(blob_ort[None])
            outs = self._ort_session.run(None, {self._ort_input_name: blob_ort})
        else:
            blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (imgsz, imgsz), swapRB=True)
            self.net.setInput(blob)
            outs = self.net.forward(['output0', 'output1'])
        out0 = outs[0][0]   # (116, 8400)
        proto = outs[1][0]  # (32, 160, 160)

        preds = out0.T
        boxes = preds[:, :4]
        scores = preds[:, 4:84]
        coeffs = preds[:, 84:]

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        confirmed_classes = set(class_ids[max_scores >= conf])

        # Valid mask: primary objects >= conf, OR co-occurring instances of confirmed classes >= 0.15
        valid_mask = (max_scores >= conf)
        for c in confirmed_classes:
            valid_mask |= (class_ids == c) & (scores[:, c] >= 0.15)
            # If cell phone is confirmed, also catch handheld devices (e.g. remote 65) that belong to series
            if c == 67:
                valid_mask |= (class_ids == 65) & (scores[:, 65] >= 0.15)

        if not np.any(valid_mask):
            return []

        boxes = boxes[valid_mask]
        coeffs = coeffs[valid_mask]
        sel_scores = max_scores[valid_mask]
        sel_classes = class_ids[valid_mask]

        if 67 in confirmed_classes:
            sel_classes[sel_classes == 65] = 67

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2

        cv_boxes = [[int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])] for i in range(len(boxes))]
        indices = cv2.dnn.NMSBoxes(cv_boxes, sel_scores.tolist(), 0.15, iou)
        if len(indices) == 0:
            return []

        proto_flat = proto.reshape(32, 160 * 160)
        scale_160 = 160 / imgsz
        my1 = int(round(dy * scale_160))
        my2 = int(round((dy + nh) * scale_160))
        mx1 = int(round(dx * scale_160))
        mx2 = int(round((dx + nw) * scale_160))

        results = []
        for idx in indices:
            i = int(idx)
            coeff = coeffs[i]
            c_mask = 1 / (1 + np.exp(-np.dot(coeff, proto_flat))).reshape(160, 160)

            bx1 = int(np.clip(x1[i] * scale_160, 0, 160))
            by1 = int(np.clip(y1[i] * scale_160, 0, 160))
            bx2 = int(np.clip((x1[i] + bw[i]) * scale_160, 0, 160))
            by2 = int(np.clip((y1[i] + bh[i]) * scale_160, 0, 160))

            cropped_160 = np.zeros_like(c_mask)
            cropped_160[by1:by2, bx1:bx2] = c_mask[by1:by2, bx1:bx2]

            content_mask = cropped_160[my1:my2, mx1:mx2]
            full_mask = cv2.resize((content_mask > 0.5).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            results.append({
                "mask": full_mask > 0,
                "cls_idx": int(sel_classes[i]),
                "conf": float(sel_scores[i])
            })
        return results


def _resolve_weight_path(weights_path: str) -> Optional[str]:
    """Resolves a model weight file across CWD, package root, and active weights directory."""
    import os
    from .config import get_weights_root
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    w_root = get_weights_root()
    base = os.path.basename(weights_path)
    candidates = [
        weights_path,
        os.path.join(w_root, base),
        os.path.join(pkg_root, base),
        os.path.join(pkg_root, weights_path),
        os.path.join(os.getcwd(), base),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def _configure_ultralytics_weights_dir() -> str:
    """Points Ultralytics' weights directory to our portable weights root so fresh clones
    and global CLI installs cache YOLO/CLIP checkpoints in one clean location."""
    import os
    import sys
    from pathlib import Path
    from .config import get_weights_root

    pkg_weights = Path(get_weights_root())
    pkg_weights.mkdir(parents=True, exist_ok=True)
    try:
        import ultralytics
        import ultralytics.utils as u_utils
        ultralytics.settings.update({"weights_dir": str(pkg_weights)})
        u_utils.WEIGHTS_DIR = pkg_weights
        for mod_name in ("ultralytics.nn.text_model", "ultralytics.utils.downloads"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "WEIGHTS_DIR"):
                setattr(mod, "WEIGHTS_DIR", pkg_weights)
    except Exception:
        pass
    return str(pkg_weights)


def get_yolo_model(model_path: str = "yolo11n-seg.pt") -> Any:
    """Lazy singleton loader for YOLO11n-seg (prefers cached ONNX runtime for fast cold startup)."""
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is None:
        onnx_name = model_path.replace(".pt", ".onnx") if model_path.endswith(".pt") else "yolo11n-seg.onnx"
        resolved_onnx = _resolve_weight_path(onnx_name) or _resolve_weight_path("yolo11n-seg.onnx")
        if resolved_onnx:
            try:
                _CACHED_YOLO_MODEL = OpenCVDNNYoloSeg(resolved_onnx)
                return _CACHED_YOLO_MODEL
            except Exception as exc_onnx:
                logger.warning("ONNX YOLO11n-seg load failed (%s); falling back to PyTorch weights.", exc_onnx)
                _CACHED_YOLO_MODEL = None

        w_root = _configure_ultralytics_weights_dir()
        resolved_pt = _resolve_weight_path(model_path)
        target_pt = resolved_pt or os.path.join(w_root, os.path.basename(model_path))
        try:
            from ultralytics import YOLO
            # Pass bare filename when missing so Ultralytics auto-downloads directly into weights_dir
            _CACHED_YOLO_MODEL = YOLO(resolved_pt if resolved_pt else os.path.basename(model_path))
        except Exception as exc:
            logger.warning("Failed to load Ultralytics YOLO model '%s' (%s).", model_path, exc)
            _CACHED_YOLO_MODEL = None
            if not resolved_pt and os.path.exists(target_pt):
                try:
                    from ultralytics import YOLO
                    _CACHED_YOLO_MODEL = YOLO(target_pt)
                except Exception as exc2:
                    logger.warning("Fallback YOLO load '%s' failed (%s).", target_pt, exc2)
                    _CACHED_YOLO_MODEL = None
    return _CACHED_YOLO_MODEL


_CACHED_STRUCT_MODEL = None


def _set_world_classes_cached(world_model: Any, classes: List[str]) -> None:
    """Sets YOLO-World classes using cached CLIP text features when available to skip loading ViT-B-32.pt (~338MB)."""
    import hashlib
    from .config import get_weights_root

    sig = hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()[:16]
    cache_path = os.path.join(get_weights_root(), f"world_txt_{sig}.npz")

    if os.path.exists(cache_path):
        try:
            import torch
            data = np.load(cache_path, allow_pickle=False)
            cached_cls = data["classes"].tolist()
            if list(cached_cls) == list(classes):
                feats = np.ascontiguousarray(data["txt_feats"], dtype=np.float32)
                dev = getattr(world_model, "device", "cpu")
                world_model.model.txt_feats = torch.from_numpy(feats).to(dev)
                world_model.model.model[-1].nc = len(classes)
                world_model.model.names = list(classes)
                world_model.predictor = None
                return
        except Exception as exc:
            logger.debug("YOLO-World text feature cache miss (%s); computing via CLIP.", exc)

    world_model.set_classes(classes)
    try:
        txt_feats = getattr(world_model.model, "txt_feats", None)
        if txt_feats is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.{os.getpid()}.tmp"
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(
                    fh,
                    classes=np.array(list(classes)),
                    txt_feats=txt_feats.detach().cpu().numpy().astype(np.float32),
                )
            os.replace(tmp_path, cache_path)
    except Exception:
        pass


def get_structural_model(model_path: str = "yolov8s-worldv2.pt") -> Any:
    """
    Lazy singleton loader for secondary open-vocabulary structural sensor (YOLO-World / YOLOE).
    Auto-downloads into weights root if weights are not yet cached.
    """
    global _CACHED_STRUCT_MODEL
    if _CACHED_STRUCT_MODEL is None:
        _configure_ultralytics_weights_dir()
        try:
            from ultralytics import YOLO
            resolved = (
                _resolve_weight_path(model_path)
                or _resolve_weight_path("yolov8s-worldv2.pt")
                or _resolve_weight_path("yoloe-v8s-seg.pt")
            )
            m = YOLO(resolved if resolved else os.path.basename(model_path))
            if hasattr(m, "set_classes"):
                _set_world_classes_cached(m, [
                    "storefront", "door", "window", "stairs", "signboard", "wall", "pavement"
                ])
            _CACHED_STRUCT_MODEL = m
        except Exception:
            _CACHED_STRUCT_MODEL = None
    return _CACHED_STRUCT_MODEL


class YoloSemanticExtractor(BaseSemanticExtractor):
    """
    IMGSTD::V04 Multimodal Semantic Extractor.
    Integrates:
      1. YOLO11n-seg for real-world concrete object instance segmentation.
      2. Open-Vocabulary Architectural Structural Detection (0100-0108).
      3. Fast Scene OCR ([TXT] Layer) for English and Arabic typography.
      4. Honest Background Modeling (0900-0906) with zero abstract hallucination.
      5. Spatial Hierarchy & Containment Reflection Reasoning.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        conf: float = 0.30,
        iou: float = 0.5,
        imgsz: int = 640,
        use_yolo: bool = True,
        use_ocr: bool = True,
        use_structural_world: bool = False,
        max_regions: int = 24
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.use_yolo = use_yolo
        self.use_ocr = use_ocr
        self.use_structural_world = use_structural_world
        self.max_regions = max_regions
        self.ocr = FastSceneOCR(use_ocr=use_ocr)
        self.last_texts: List[TextDescriptor] = []

    def extract(self, image: np.ndarray) -> Tuple[List[RegionDescriptor], np.ndarray, np.ndarray, List[TextDescriptor]]:
        h_orig, w_orig = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        total_pixels = h_orig * w_orig

        label_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
        color_lut = np.zeros((256, c), dtype=np.uint8)
        results: List[RegionDescriptor] = []
        next_label_id = 1

        # -------------------------------------------------------------
        # 1. Fast Scene OCR ([TXT] Layer)
        # -------------------------------------------------------------
        texts: List[TextDescriptor] = self.ocr.extract(image)
        self.last_texts = texts

        # -------------------------------------------------------------
        # 2. Run YOLO11n-seg for Object Instance Detection
        # -------------------------------------------------------------
        raw_instances = []
        if self.use_yolo:
            model = get_yolo_model(self.model_path)
            if model is not None:
                try:
                    image_pred = image[:, :, :3] if (c == 4) else image
                    if isinstance(model, OpenCVDNNYoloSeg):
                        raw_instances = model.predict(image_pred, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
                    else:
                        preds = model.predict(
                            image_pred,
                            imgsz=self.imgsz,
                            conf=self.conf,
                            iou=self.iou,
                            verbose=False
                        )
                        if preds and len(preds) > 0 and preds[0].masks is not None:
                            boxes = preds[0].boxes
                            masks_data = preds[0].masks.data.cpu().numpy()
                            cls_indices = boxes.cls.cpu().numpy().astype(int)
                            conf_scores = boxes.conf.cpu().numpy()

                            for i in range(len(cls_indices)):
                                m_raw = masks_data[i]
                                m_full = cv2.resize(
                                    (m_raw > 0.5).astype(np.uint8),
                                    (w_orig, h_orig),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                raw_instances.append({
                                    "mask": m_full > 0,
                                    "cls_idx": int(cls_indices[i]),
                                    "conf": float(conf_scores[i])
                                })
                except Exception as exc:
                    logger.warning("YOLO11n-seg inference failed (%s); falling back cleanly.", exc)
                    raw_instances = []

        # -------------------------------------------------------------
        # 3. Open-Vocabulary Architectural Structural Detection
        # -------------------------------------------------------------
        structural_candidates = []
        if self.use_structural_world:
            # Secondary Structural/Architectural Sensor (opt-in)
            struct_model = get_structural_model()
            if struct_model is not None:
                try:
                    image_pred = image[:, :, :3] if (c == 4) else image
                    s_preds = struct_model.predict(image_pred, imgsz=self.imgsz, conf=0.25, verbose=False)
                    if s_preds and len(s_preds) > 0 and s_preds[0].boxes is not None:
                        s_boxes = s_preds[0].boxes.xyxy.cpu().numpy()
                        s_cls = s_preds[0].boxes.cls.cpu().numpy().astype(int)
                        s_conf = s_preds[0].boxes.conf.cpu().numpy()
                        world_map = {
                            0: "0101",  # storefront
                            1: "0102",  # door
                            2: "0103",  # window
                            3: "0104",  # stairs
                            4: "0105",  # signboard
                            5: "0106",  # wall
                            6: "0107",  # pavement
                        }
                        for b_idx in range(len(s_boxes)):
                            bx1, by1, bx2, by2 = s_boxes[b_idx].astype(int)
                            bx1, by1 = max(0, bx1), max(0, by1)
                            bx2, by2 = min(w_orig, bx2), min(h_orig, by2)
                            if bx2 > bx1 and by2 > by1:
                                sm = np.zeros((h_orig, w_orig), dtype=bool)
                                sm[by1:by2, bx1:bx2] = True
                                c_id = world_map.get(s_cls[b_idx], "0101")
                                s_area = (bx2 - bx1) * (by2 - by1)
                                s_cov = (s_area / total_pixels) * 100.0
                                structural_candidates.append({
                                    "mask": sm,
                                    "class_id": c_id,
                                    "conf": float(s_conf[b_idx]),
                                    "prominence": "PRIMARY" if s_cov >= 5.0 else ("SECONDARY" if s_cov >= 1.5 else "MINOR_DETAIL")
                                })
                except Exception:
                    structural_candidates = []

            # Architectural structures are populated strictly by actual structural model or explicit mock hooks
            if not structural_candidates:
                structural_candidates = []

        # Register structural candidates
        structural_regions: List[RegionDescriptor] = []
        for st in structural_candidates:
            s_mask = st["mask"]
            s_area = int(np.count_nonzero(s_mask))
            if s_area < 16:
                continue

            label_id = next_label_id
            next_label_id += 1
            label_map[s_mask] = label_id

            y_idx, x_idx = np.where(s_mask)
            bx = int(np.min(x_idx))
            by = int(np.min(y_idx))
            bw = int(np.max(x_idx) - bx + 1)
            bh = int(np.max(y_idx) - by + 1)

            pixels_in_mask = image[s_mask]
            dom_col = np.median(pixels_in_mask, axis=0).astype(int).tolist()[:c]
            color_lut[label_id] = dom_col

            cov_pct = round((s_area / total_pixels) * 100.0, 2)
            cls_id = st["class_id"]
            conf_val = round(float(st.get("conf", 0.90)), 2)
            s_role = "PRIMARY" if cov_pct >= 5.0 else ("SECONDARY" if cov_pct >= 1.5 else "MINOR_DETAIL")

            s_desc = RegionDescriptor(
                region_id=f"{label_id:03d}",
                class_id=cls_id,
                bbox=(bx, by, bw, bh),
                dominant_color=dom_col,
                label_id=label_id,
                area=s_area,
                coverage_pct=cov_pct,
                kind="STRUCT",
                conf=conf_val,
                prominence=s_role,
                role=s_role
            )
            structural_regions.append(s_desc)
            results.append(s_desc)

        # -------------------------------------------------------------
        # 4. Assign Object Instances & Spatial Containment Reasoning
        # -------------------------------------------------------------
        raw_instances.sort(key=lambda inst: int(np.count_nonzero(inst["mask"])), reverse=True)

        for inst in raw_instances[:self.max_regions]:
            mask = inst["mask"]
            inst_area = int(np.count_nonzero(mask))
            if inst_area < max(16, int(total_pixels * 0.003)):
                continue

            label_id = next_label_id
            next_label_id += 1

            label_map[mask] = label_id

            y_indices, x_indices = np.where(mask)
            bx = int(np.min(x_indices))
            by = int(np.min(y_indices))
            bw = int(np.max(x_indices) - bx + 1)
            bh = int(np.max(y_indices) - by + 1)

            pixels_in_mask = image[mask]
            dom_col = np.median(pixels_in_mask, axis=0).astype(int).tolist()[:c]
            color_lut[label_id] = dom_col

            cov_pct = round((inst_area / total_pixels) * 100.0, 2)
            cls_idx = inst.get("cls_idx", 0)
            cls_id = inst.get("cls_id") or COCO_CLASS_ID_MAP.get(cls_idx, f"{cls_idx+1:04d}")
            conf_val = round(float(inst.get("conf", 1.0)), 2)

            # Prominence classification (ROLE)
            # PRIMARY: COV >= 5.0%
            # SECONDARY: 1.5% <= COV < 5.0%
            # MINOR_DETAIL: COV < 1.5%
            if cov_pct >= 5.0:
                role = "PRIMARY"
            elif cov_pct >= 1.5:
                role = "SECONDARY"
            else:
                role = "MINOR_DETAIL"

            is_refl = False
            likely_refl = False
            parent_struct = None
            contained_in = None

            # Spatial Containment: IoA (Intersection over Area >= 0.75) inside Door (0102) or Window (0103)
            for st in structural_regions:
                if st.class_id in ("0102", "0103"):
                    ox, oy, ow, oh = st.bbox
                    ix1 = max(bx, ox)
                    iy1 = max(by, oy)
                    ix2 = min(bx + bw, ox + ow)
                    iy2 = min(by + bh, oy + oh)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter_area = iw * ih
                    ioa = inter_area / max(1, bw * bh)
                    if ioa >= 0.75:
                        is_refl = True
                        likely_refl = True
                        role = "MINOR_DETAIL"
                        st_name = "Door / Entrance" if st.class_id == "0102" else "Window / Glass Panel"
                        contained_in = f"REG:{st.region_id} ({st_name})"
                        parent_struct = f"{st.class_id} ({st_name})"
                        break

            results.append(RegionDescriptor(
                region_id=f"{label_id:03d}",
                class_id=cls_id,
                bbox=(bx, by, bw, bh),
                dominant_color=dom_col,
                label_id=label_id,
                area=inst_area,
                coverage_pct=cov_pct,
                kind="OBJ",
                conf=conf_val,
                prominence=role,
                role=role,
                is_reflection=is_refl,
                likely_reflection_or_inset=likely_refl,
                parent_structure=parent_struct,
                contained_in=contained_in
            ))

        # -------------------------------------------------------------
        # 2. Honest Background Modeling (for all pixels where label_map == 0)
        # -------------------------------------------------------------
        bg_mask = (label_map == 0)
        bg_pixel_count = int(np.count_nonzero(bg_mask))

        if bg_pixel_count > 0:
            # If remaining background is tiny (<1%) and objects exist, fold into object
            if bg_pixel_count < int(total_pixels * 0.01) and results:
                # Assign to nearest label
                first_label = results[0].label_id
                label_map[bg_mask] = first_label
            else:
                # Downscale background for rapid connected components / clustering
                sw = max(16, min(160, w_orig))
                sh = max(16, min(160, h_orig))

                small_img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
                small_bg = cv2.resize(bg_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

                # Quantize colors into bins of 48
                q_small = (small_img // 48) * 48
                c_ch = min(3, c)
                code = (q_small[:, :, 0].astype(np.int32) * 65536 +
                        (q_small[:, :, 1].astype(np.int32) * 256 if c_ch > 1 else 0) +
                        (q_small[:, :, 2].astype(np.int32) if c_ch > 2 else 0))

                # Zero out non-background pixels in code
                code[~small_bg] = -1

                u_codes, counts = np.unique(code[code >= 0], return_counts=True)
                sorted_idx = np.argsort(-counts)
                u_codes = u_codes[sorted_idx]

                remaining_bg = bg_mask.copy()
                min_bg_cluster = max(16, int(total_pixels * 0.01))

                for code_val in u_codes:
                    if next_label_id >= 255:
                        break
                    cluster_small = ((code == code_val) & small_bg).astype(np.uint8)
                    if cluster_small.sum() < max(4, int(sw * sh * 0.01)):
                        continue

                    cluster_full = cv2.resize(cluster_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST) > 0
                    cluster_full = cluster_full & remaining_bg

                    cluster_area = int(np.count_nonzero(cluster_full))
                    if cluster_area < min_bg_cluster:
                        continue

                    label_id = next_label_id
                    next_label_id += 1

                    label_map[cluster_full] = label_id
                    remaining_bg[cluster_full] = False

                    # Bounding box
                    y_idx, x_idx = np.where(cluster_full)
                    bx = int(np.min(x_idx))
                    by = int(np.min(y_idx))
                    bw = int(np.max(x_idx) - bx + 1)
                    bh = int(np.max(y_idx) - by + 1)

                    # Dominant color
                    dom_col = np.median(image[cluster_full], axis=0).astype(int).tolist()[:c]
                    color_lut[label_id] = dom_col

                    cov_pct = round((cluster_area / total_pixels) * 100.0, 2)

                    # Classify background honestly (0900-0906)
                    r = dom_col[0]
                    g = dom_col[1] if c > 1 else r
                    b = dom_col[2] if c > 2 else r
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    yc = (by + bh / 2.0) / max(1, h_orig)

                    # Color standard deviation inside cluster (measures texture vs flat)
                    pix_std = float(np.std(image[cluster_full], axis=0).mean())

                    if lum < 45:
                        cls_id = "0901"  # Dark / Shadow Background
                    elif lum > 210:
                        cls_id = "0902"  # Bright / Highlight Background
                    elif yc < 0.35 and (b > r + 15 or lum > 160):
                        cls_id = "0904"  # Sky / Upper Backdrop
                    elif pix_std > 20:
                        cls_id = "0903"  # Gradient / Abstract Texture
                    else:
                        cls_id = "0900"  # Unclassified Background / Surface

                    results.append(RegionDescriptor(
                        region_id=f"{label_id:03d}",
                        class_id=cls_id,
                        bbox=(bx, by, bw, bh),
                        dominant_color=dom_col,
                        label_id=label_id,
                        area=cluster_area,
                        coverage_pct=cov_pct,
                        kind="BG",
                        conf=1.00,
                        prominence="BACKGROUND",
                        role="BACKGROUND"
                    ))

                # Any lingering unassigned background pixels
                if np.any(remaining_bg):
                    unassigned_y, unassigned_x = np.where(remaining_bg)
                    if len(results) > 0:
                        # Map to the first background region (or first region)
                        bg_regions = [r for r in results if r.kind == "BG"]
                        fallback_label = bg_regions[0].label_id if bg_regions else results[0].label_id
                        label_map[remaining_bg] = fallback_label
                    else:
                        # Whole image is single background
                        label_id = 1
                        label_map[:] = label_id
                        dom_col = np.median(image.reshape(-1, c), axis=0).astype(int).tolist()[:c]
                        color_lut[label_id] = dom_col
                        results.append(RegionDescriptor(
                            region_id="001",
                            class_id="0900",
                            bbox=(0, 0, w_orig, h_orig),
                            dominant_color=dom_col,
                            label_id=1,
                            area=total_pixels,
                            coverage_pct=100.0,
                            kind="BG",
                            conf=1.00,
                            prominence="BACKGROUND",
                            role="BACKGROUND"
                        ))

        # Sort all results so objects come first, then background, or sorted by area
        return results, label_map, color_lut, texts


class FastColorSegmenter(BaseSemanticExtractor):
    """
    Deterministic, ultra-fast color-quantized connected-components extractor (V01/V02).
    Maintained for lightweight or non-YOLO workflows.
    """

    def __init__(self, max_dim: int = 128, max_regions: int = 8, min_area_ratio: float = 0.02):
        self.max_dim = max_dim
        self.max_regions = max_regions
        self.min_area_ratio = min_area_ratio

    def extract(self, image: np.ndarray) -> Tuple[List[RegionDescriptor], np.ndarray, np.ndarray]:
        h_orig, w_orig = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        total_pixels = h_orig * w_orig

        label_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
        color_lut = np.zeros((256, c), dtype=np.uint8)

        if total_pixels < 64:
            median_color = np.median(image.reshape(-1, c), axis=0).astype(int).tolist()[:c]
            label_map[:] = 1
            color_lut[1] = median_color
            return [RegionDescriptor(
                region_id="001",
                class_id="0900",
                bbox=(0, 0, w_orig, h_orig),
                dominant_color=median_color,
                label_id=1,
                area=total_pixels,
                coverage_pct=100.0,
                kind="BG",
                conf=1.00
            )], label_map, color_lut

        scale = min(1.0, float(self.max_dim) / max(h_orig, w_orig))
        sw, sh = max(1, int(w_orig * scale)), max(1, int(h_orig * scale))
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = small[:, :, None]

        q_small = (small // 48) * 48
        c_channels = min(3, image.shape[-1])
        code = (q_small[:, :, 0].astype(np.int32) * 65536 +
                (q_small[:, :, 1].astype(np.int32) * 256 if c_channels > 1 else 0) +
                (q_small[:, :, 2].astype(np.int32) if c_channels > 2 else 0))

        u_codes, counts = np.unique(code, return_counts=True)
        top_indices = np.argsort(-counts)[:self.max_regions]
        top_codes = u_codes[top_indices]

        regions_raw = []
        min_pixels_small = int(sw * sh * self.min_area_ratio)

        for c_val in top_codes:
            mask_small = (code == c_val).astype(np.uint8)
            if mask_small.sum() < min_pixels_small:
                continue

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_small, connectivity=8)
            for i in range(1, num_labels):
                comp_area = stats[i, cv2.CC_STAT_AREA]
                if comp_area < min_pixels_small:
                    continue

                comp_mask_small = (labels == i)
                pixels_small = small[comp_mask_small]
                median_rgb = np.median(pixels_small, axis=0).astype(int).tolist()

                sx = int(stats[i, cv2.CC_STAT_LEFT])
                sy = int(stats[i, cv2.CC_STAT_TOP])
                sw_box = int(stats[i, cv2.CC_STAT_WIDTH])
                sh_box = int(stats[i, cv2.CC_STAT_HEIGHT])

                bx = int(sx / scale)
                by = int(sy / scale)
                bw = min(w_orig - bx, int(sw_box / scale))
                bh = min(h_orig - by, int(sh_box / scale))

                regions_raw.append({
                    "bbox": (bx, by, bw, bh),
                    "dominant_color": median_rgb,
                    "comp_mask_small": comp_mask_small,
                    "area": int(comp_area / (scale * scale)),
                })

        regions_raw.sort(key=lambda r: r["area"], reverse=True)

        if not regions_raw:
            median_color = np.median(image.reshape(-1, c), axis=0).astype(int).tolist()[:c]
            label_map[:] = 1
            color_lut[1] = median_color
            return [RegionDescriptor(
                region_id="001",
                class_id="0900",
                bbox=(0, 0, w_orig, h_orig),
                dominant_color=median_color,
                label_id=1,
                area=total_pixels,
                coverage_pct=100.0,
                kind="BG",
                conf=1.00
            )], label_map, color_lut

        label_map_small = np.zeros((sh, sw), dtype=np.uint8)
        results: List[RegionDescriptor] = []
        for idx, r in enumerate(regions_raw[:self.max_regions]):
            label_id = idx + 1
            reg_id = f"{label_id:03d}"
            cov_pct = round((r["area"] / total_pixels) * 100.0, 2)
            label_map_small[r["comp_mask_small"]] = label_id

            dom_col = r["dominant_color"][:c]
            color_lut[label_id] = dom_col

            r_val = dom_col[0]
            g_val = dom_col[1] if c > 1 else r_val
            b_val = dom_col[2] if c > 2 else r_val
            lum = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
            cls_id = "0901" if lum < 45 else ("0902" if lum > 210 else "0900")

            results.append(RegionDescriptor(
                region_id=reg_id,
                class_id=cls_id,
                bbox=r["bbox"],
                dominant_color=r["dominant_color"],
                label_id=label_id,
                area=r["area"],
                coverage_pct=cov_pct,
                kind="BG",
                conf=1.00
            ))

        label_map = cv2.resize(label_map_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        # Any 0 pixels in label map get mapped to label 1
        label_map[label_map == 0] = 1

        return results, label_map, color_lut


# =====================================================================
# Open-Vocabulary Semantic Dictionary & Extractor (YOLO-World)
# =====================================================================

# OPEN_VOCAB_MAP imported from .vocab_maps

_CACHED_WORLD_MODEL = None
_CACHED_WORLD_HYBRID_ORT: Dict[str, "ORTYoloWorld"] = {}


class ORTYoloWorld:
    """
    PyTorch-free ONNXRuntime runner for fixed-vocabulary YOLO-World hybrid models.
    Pre-bakes class text embeddings inside the ONNX graph so cold CLI runs skip
    importing torch, ultralytics, and CLIP ViT-B-32 entirely.
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def predict_boxes(
        self,
        image: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.42,
        iou: float = 0.45,
    ) -> List[Dict[str, Any]]:
        h_orig, w_orig = image.shape[:2]
        scale = min(imgsz / w_orig, imgsz / h_orig)
        nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        dx = (imgsz - nw) // 2
        dy = (imgsz - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) * (1.0 / 255.0)
        )
        out = self._sess.run(None, {self._input_name: blob})[0][0].T  # (8400, 4 + C)
        boxes = out[:, :4]
        scores = out[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        keep = max_scores >= conf
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        cls_ids = cls_ids[keep]
        max_scores = max_scores[keep]

        x1 = (boxes[:, 0] - boxes[:, 2] * 0.5 - dx) / scale
        y1 = (boxes[:, 1] - boxes[:, 3] * 0.5 - dy) / scale
        bw = boxes[:, 2] / scale
        bh = boxes[:, 3] / scale

        cv_boxes = [[float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])] for i in range(len(boxes))]
        indices = cv2.dnn.NMSBoxes(cv_boxes, max_scores.tolist(), conf, iou)
        if len(indices) == 0:
            return []

        results: List[Dict[str, Any]] = []
        for idx in np.asarray(indices).reshape(-1):
            i = int(idx)
            bx = int(max(0, min(w_orig - 1, round(x1[i]))))
            by = int(max(0, min(h_orig - 1, round(y1[i]))))
            x2 = int(max(bx + 1, min(w_orig, round(x1[i] + bw[i]))))
            y2 = int(max(by + 1, min(h_orig, round(y1[i] + bh[i]))))
            results.append({
                "xyxy": (bx, by, x2, y2),
                "cls_idx": int(cls_ids[i]),
                "conf": float(max_scores[i]),
            })
        return results


def get_hybrid_yolo_world_ort(
    classes: List[str],
    weights_path: str = "yolov8s-worldv2.pt",
) -> Optional[ORTYoloWorld]:
    """
    Returns an ONNXRuntime runner for fixed open-vocabulary classes, exporting once
    from the PyTorch YOLO-World checkpoint when `onnx` is installed.
    """
    import hashlib
    from .config import get_weights_root

    sig = hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()[:16]
    if sig in _CACHED_WORLD_HYBRID_ORT:
        return _CACHED_WORLD_HYBRID_ORT[sig]

    w_root = get_weights_root()
    tagged_onnx = os.path.join(w_root, f"yolov8s-worldv2_hybrid_{sig}.onnx")
    legacy_onnx = os.path.join(w_root, "yolov8s-worldv2_hybrid.onnx")

    # Adopt existing un-hashed hybrid ONNX if present for HYBRID_OPEN_CLASSES
    onnx_path = tagged_onnx if os.path.exists(tagged_onnx) else (
        legacy_onnx if (os.path.exists(legacy_onnx) and list(classes) == list(YoloWorldSemanticExtractor.HYBRID_OPEN_CLASSES)) else tagged_onnx
    )

    if os.path.exists(onnx_path):
        try:
            runner = ORTYoloWorld(onnx_path)
            _CACHED_WORLD_HYBRID_ORT[sig] = runner
            return runner
        except Exception as exc:
            logger.debug("Cached YOLO-World ONNX load skipped (%s).", exc)

    # Auto-export once if onnx package & PyTorch model are available
    try:
        import onnx  # noqa: F401
        wm = get_yolo_world_model(weights_path)
        if wm is not None:
            _set_world_classes_cached(wm, classes)
            exported_path = wm.export(format="onnx", imgsz=640, opset=17, simplify=False)
            if exported_path and os.path.exists(exported_path):
                os.makedirs(w_root, exist_ok=True)
                os.replace(exported_path, tagged_onnx)
                runner = ORTYoloWorld(tagged_onnx)
                _CACHED_WORLD_HYBRID_ORT[sig] = runner
                return runner
    except Exception as exc:
        logger.debug("YOLO-World ONNX auto-export skipped (%s).", exc)
    return None


def get_yolo_world_model(weights_path: str = "yolov8s-worldv2.pt"):
    """Singleton lazy loader for YOLO-World model (auto-downloads into portable weights root if missing)."""
    global _CACHED_WORLD_MODEL
    if _CACHED_WORLD_MODEL is not None:
        return _CACHED_WORLD_MODEL

    _configure_ultralytics_weights_dir()
    try:
        from ultralytics import YOLOWorld
        resolved = _resolve_weight_path(weights_path)
        model = YOLOWorld(resolved if resolved else os.path.basename(weights_path))
        _CACHED_WORLD_MODEL = model
        return _CACHED_WORLD_MODEL
    except Exception as exc:
        logger.warning("Failed to initialize YOLO-World '%s' (%s); skipping open-vocab detection.", weights_path, exc)
        return None


class YoloWorldSemanticExtractor(BaseSemanticExtractor):
    """
    Open-Vocabulary Semantic Extractor utilizing YOLO-World (yolov8s-worldv2.pt)
    with extensible open-vocabulary prompts, automatic celestial & cosmic detection,
    spatial containment reasoning, and fast multilingual OCR.
    
    Fallback chain:
        YOLO-World -> YoloSemanticExtractor (YOLO11n-seg) -> FastColorSegmenter
    """

    DEFAULT_VOCABULARY: List[str] = [
        # Outdoor, Natural & Industrial
        "desert", "sand dune", "mountain", "ocean", "sky", "street",
        "heavy machinery", "aircraft", "spaceship", "soldier",
        # Architecture & UI
        "building facade", "storefront", "glass door", "window", "stairs",
        "signboard", "wall", "room", "text banner", "logo", "screen", "chart", "diagram",
        # Standard everyday physical objects & animals
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "bench", "bird", "cat", "dog", "animal", "rodent", "chair", "couch", "potted plant",
        "bed", "dining table", "cell phone", "laptop", "book", "clock"
    ]

    # Compact non-COCO open-vocabulary targets used cleanly alongside YOLO11n-seg without class dilution
    HYBRID_OPEN_CLASSES: List[str] = [
        "heavy machinery", "spaceship", "soldier", "rodent", "chart", "diagram"
    ]

    def __init__(
        self,
        model_path: str = "yolov8s-worldv2.pt",
        conf: float = 0.30,
        iou: float = 0.45,
        imgsz: int = 640,
        custom_classes: Optional[List[str]] = None,
        use_ocr: bool = True
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._custom_classes_provided = (custom_classes is not None)
        self.active_classes = list(custom_classes) if custom_classes else list(self.DEFAULT_VOCABULARY)
        self.use_ocr = use_ocr
        self.ocr = FastSceneOCR(use_ocr=use_ocr)
        self.fallback_extractor = YoloSemanticExtractor(use_ocr=use_ocr)
        self.last_texts: List[TextDescriptor] = []
        self._model_classes_set = False

    def set_classes(self, classes: List[str]):
        """Dynamically updates active open-vocabulary prompt at runtime."""
        self.active_classes = list(classes)
        self._custom_classes_provided = True
        model = get_yolo_world_model(self.model_path)
        if model is not None:
            try:
                _set_world_classes_cached(model, self.active_classes)
                self._model_classes_set = True
            except Exception:
                pass

    @staticmethod
    def _box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
        return inter / max(1, union)

    def extract(
        self,
        image: np.ndarray
    ) -> Tuple[List[RegionDescriptor], np.ndarray, np.ndarray, List[TextDescriptor]]:
        """
        Extracts open-vocabulary semantic regions from image with anti-hallucination guarantees.
        """
        from concurrent.futures import ThreadPoolExecutor

        h_orig, w_orig = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        total_pixels = h_orig * w_orig

        # Run RapidOCR asynchronously while celestial check + YOLO11n-seg execute
        ocr_executor = ThreadPoolExecutor(max_workers=1)
        ocr_future = ocr_executor.submit(self.ocr.extract, image)

        def _collect_ocr(fallback: Optional[List[TextDescriptor]] = None) -> List[TextDescriptor]:
            try:
                res = ocr_future.result()
                return res if (res or fallback is None) else fallback
            except Exception as exc:
                logger.warning("Async OCR extraction failed (%s).", exc)
                return fallback or []
            finally:
                ocr_executor.shutdown(wait=False)

        # 2. Celestial & Deep Space Analysis (Strict void + circularity, letterbox-aware)
        content_slice = image
        if h_orig >= 64:
            row_means = image.mean(axis=(1, 2)) if c == 3 else image.mean(axis=1)
            non_black = np.where(row_means >= 12.0)[0]
            if len(non_black) > int(h_orig * 0.40):
                y1, y2 = int(non_black[0]), int(non_black[-1]) + 1
                if y2 - y1 >= int(h_orig * 0.40):
                    content_slice = image[y1:y2, :]

        pitch_black = (content_slice < 15).all(axis=-1) if c >= 3 else (content_slice < 15)
        dark_pct = float(np.mean(pitch_black) * 100.0)
        is_deep_space = dark_pct >= 60.0

        celestial_regions = []
        if is_deep_space:
            # Check for true circular celestial sphere (e.g. Planet Earth, Moon); handle RGB, RGBA & Gray cleanly
            if c == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif c == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                gray = image
            thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                area = cv2.contourArea(cnt)
                perim = cv2.arcLength(cnt, True)
                circularity = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
                if area >= (total_pixels * 0.04) and circularity >= 0.78:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = bw / float(max(1, bh))
                    if 0.78 <= aspect_ratio <= 1.28:
                        mask_c = np.zeros((h_orig, w_orig), dtype=np.uint8)
                        cv2.drawContours(mask_c, [cnt], -1, 255, -1)
                        # Verify exterior void is truly dark cosmic space (not a dim indoor backdrop)
                        outside_gray = gray[mask_c == 0]
                        if outside_gray.size > 0 and float(np.mean(outside_gray)) > 20.0:
                            continue
                        cov_pct = round((area / total_pixels) * 100.0, 2)
                        pts = image[mask_c > 0]
                        dom_col = np.median(pts, axis=0).astype(int).tolist()[:c] if len(pts) > 0 else [50, 100, 150]

                        # Compute embedding
                        emb = RegionEmbeddingGenerator.compute(image, (bx, by, bw, bh), mask_c > 0)

                        celestial_regions.append(RegionDescriptor(
                            region_id="001",
                            class_id="0201",
                            class_name="Planet Earth / Globe (Celestial)",
                            category="Celestial & Space",
                            bbox=(bx, by, bw, bh),
                            dominant_color=dom_col,
                            label_id=1,
                            area=int(area),
                            coverage_pct=cov_pct,
                            kind="OBJ",
                            conf=0.96,
                            prominence="PRIMARY",
                            role="PRIMARY",
                            embedding=emb
                        ))
                        break

        # If celestial sphere detected, construct honest space background and return immediately
        if celestial_regions:
            label_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
            color_lut = np.zeros((256, c), dtype=np.uint8)

            # Draw celestial sphere
            cr = celestial_regions[0]
            mask_c = np.zeros((h_orig, w_orig), dtype=np.uint8)
            bx, by, bw, bh = cr.bbox
            cv2.ellipse(
                mask_c,
                (bx + bw // 2, by + bh // 2),
                (bw // 2, bh // 2),
                0, 0, 360, 255, -1
            )
            label_map[mask_c > 0] = 1
            color_lut[1] = cr.dominant_color[:c]

            # Space background
            bg_mask = (label_map == 0)
            label_map[bg_mask] = 2
            color_lut[2] = [5, 5, 10][:c]

            bg_emb = RegionEmbeddingGenerator.compute(image, (0, 0, w_orig, h_orig), bg_mask)
            bg_region = RegionDescriptor(
                region_id="002",
                class_id="0206",
                class_name="Outer Space / Deep Space (Celestial)",
                category="Celestial & Space",
                bbox=(0, 0, w_orig, h_orig),
                dominant_color=[5, 5, 10][:c],
                label_id=2,
                area=int(np.count_nonzero(bg_mask)),
                coverage_pct=round(float(np.mean(bg_mask) * 100.0), 2),
                kind="BG",
                conf=1.00,
                prominence="BACKGROUND",
                role="BACKGROUND",
                embedding=bg_emb
            )
            texts = _collect_ocr()
            self.last_texts = texts
            return [cr, bg_region], label_map, color_lut, texts

        image_pred = image[:, :, :3] if (image.ndim == 3 and image.shape[2] == 4) else image

        # 3. Hybrid Inference:
        #    - Standard scenes: YOLO11n-seg (true pixel masks + high COCO recall) + non-COCO open-vocab extras.
        #    - Explicit custom_classes: YOLO-World directly on active_classes.
        candidates: List[Dict[str, Any]] = []

        if not self._custom_classes_provided:
            yolo_seg = get_yolo_model("yolo11n-seg.pt")
            coco_conf = min(0.32, max(0.25, float(self.conf)))
            if yolo_seg is not None:
                try:
                    if isinstance(yolo_seg, OpenCVDNNYoloSeg):
                        seg_insts = yolo_seg.predict(image_pred, imgsz=self.imgsz, conf=coco_conf, iou=self.iou)
                        for inst in seg_insts:
                            m = inst["mask"]
                            ys, xs = np.where(m)
                            if ys.size == 0:
                                continue
                            bx, by = int(np.min(xs)), int(np.min(ys))
                            bw, bh = int(np.max(xs) - bx + 1), int(np.max(ys) - by + 1)
                            c_idx = int(inst.get("cls_idx", 0))
                            cls_id = COCO_CLASS_ID_MAP.get(c_idx, f"{c_idx + 1:04d}")
                            raw_nm = COCO_CLASS_NAMES.get(c_idx, f"object_{c_idx}")
                            v_info = OPEN_VOCAB_MAP.get(raw_nm)
                            candidates.append({
                                "mask": m, "bbox": (bx, by, bw, bh), "area": int(np.count_nonzero(m)),
                                "cls_id": v_info[0] if v_info else cls_id,
                                "raw_name": raw_nm,
                                "cls_name": v_info[1] if v_info else raw_nm.title(),
                                "category": v_info[2] if v_info else None,
                                "conf": float(inst.get("conf", 0.5)),
                                "source": "coco_seg"
                            })
                    else:
                        seg_preds = yolo_seg.predict(image_pred, imgsz=self.imgsz, conf=coco_conf, iou=self.iou, agnostic_nms=True, verbose=False)
                        if seg_preds and len(seg_preds) > 0 and seg_preds[0].boxes is not None and len(seg_preds[0].boxes) > 0:
                            boxes = seg_preds[0].boxes
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls_indices = boxes.cls.cpu().numpy().astype(int)
                            conf_scores = boxes.conf.cpu().numpy()
                            masks_data = seg_preds[0].masks.data.cpu().numpy() if seg_preds[0].masks is not None else None
                            names_map = getattr(yolo_seg, "names", {})

                            for i in range(len(cls_indices)):
                                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                                bx = max(0, min(w_orig - 1, x1))
                                by = max(0, min(h_orig - 1, y1))
                                bw = max(1, min(w_orig - bx, x2 - x1))
                                bh = max(1, min(h_orig - by, y2 - y1))
                                if masks_data is not None and i < len(masks_data):
                                    m = cv2.resize((masks_data[i] > 0.5).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST) > 0
                                else:
                                    m = np.zeros((h_orig, w_orig), dtype=bool)
                                    m[by:by + bh, bx:bx + bw] = True
                                area = int(np.count_nonzero(m))
                                if area < 16:
                                    continue
                                c_idx = int(cls_indices[i])
                                cls_id = COCO_CLASS_ID_MAP.get(c_idx, f"{c_idx + 1:04d}")
                                raw_nm = str(names_map.get(c_idx, f"object_{c_idx}")).lower()
                                v_info = OPEN_VOCAB_MAP.get(raw_nm)
                                candidates.append({
                                    "mask": m, "bbox": (bx, by, bw, bh), "area": area,
                                    "cls_id": v_info[0] if v_info else cls_id,
                                    "raw_name": raw_nm,
                                    "cls_name": v_info[1] if v_info else raw_nm.title(),
                                    "category": v_info[2] if v_info else None,
                                    "conf": float(conf_scores[i]),
                                    "source": "coco_seg"
                                })
                except Exception as exc:
                    logger.warning("YOLO11n-seg hybrid inference failed (%s).", exc)

            # Targeted non-COCO open-vocabulary supplement (prefers PyTorch-free ONNXRuntime cache)
            ow_conf = max(0.42, float(self.conf))
            ow_dets: List[Tuple[Tuple[int, int, int, int], int, float]] = []
            ort_world = get_hybrid_yolo_world_ort(self.HYBRID_OPEN_CLASSES, self.model_path)
            if ort_world is not None:
                try:
                    for det in ort_world.predict_boxes(image_pred, imgsz=self.imgsz, conf=ow_conf, iou=self.iou):
                        ow_dets.append((det["xyxy"], det["cls_idx"], det["conf"]))
                except Exception as exc:
                    logger.warning("YOLO-World ONNX inference failed (%s); falling back to PyTorch.", exc)
                    ow_dets = []
                    ort_world = None

            if ort_world is None:
                world_model = get_yolo_world_model(self.model_path)
                if world_model is not None:
                    try:
                        if not self._model_classes_set or self.active_classes != self.HYBRID_OPEN_CLASSES:
                            _set_world_classes_cached(world_model, self.HYBRID_OPEN_CLASSES)
                            self.active_classes = list(self.HYBRID_OPEN_CLASSES)
                            self._model_classes_set = True

                        w_preds = world_model.predict(image_pred, imgsz=self.imgsz, conf=ow_conf, iou=self.iou, agnostic_nms=True, verbose=False)
                        if w_preds and len(w_preds) > 0 and w_preds[0].boxes is not None and len(w_preds[0].boxes) > 0:
                            boxes = w_preds[0].boxes
                            xyxy = boxes.xyxy.cpu().numpy()
                            cls_indices = boxes.cls.cpu().numpy().astype(int)
                            conf_scores = boxes.conf.cpu().numpy()
                            for i in range(len(cls_indices)):
                                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                                ow_dets.append(((x1, y1, x2, y2), int(cls_indices[i]), float(conf_scores[i])))
                    except Exception as exc:
                        logger.warning("YOLO-World hybrid inference failed (%s).", exc)

            for (x1, y1, x2, y2), c_idx, c_score in ow_dets:
                bx = max(0, min(w_orig - 1, x1))
                by = max(0, min(h_orig - 1, y1))
                bw = max(1, min(w_orig - bx, x2 - x1))
                bh = max(1, min(h_orig - by, y2 - y1))
                ow_box = (bx, by, bw, bh)
                raw_nm = self.HYBRID_OPEN_CLASSES[c_idx] if c_idx < len(self.HYBRID_OPEN_CLASSES) else "object"
                v_info = OPEN_VOCAB_MAP.get(raw_nm.lower(), ("0899", raw_nm.title(), "General"))

                # If 'rodent' overlaps a low-conf COCO bird/cat/dog/bear, refine label while preserving pixel mask
                refined = False
                for cand in candidates:
                    iou_val = self._box_iou(ow_box, cand["bbox"])
                    if raw_nm == "rodent" and cand["cls_id"] in ("0016", "0017", "0018", "0023") and cand["conf"] < 0.72 and iou_val >= 0.35:
                        cand["cls_id"], cand["cls_name"], cand["category"] = v_info
                        cand["raw_name"] = raw_nm
                        refined = True
                        break
                if refined:
                    continue

                # Skip open-vocab detections overlapping COCO instances (or humans wearing gear -> Soldier refinement stays if no high-conf COCO overlap)
                if any(self._box_iou(ow_box, c["bbox"]) >= 0.30 for c in candidates):
                    continue

                m = np.zeros((h_orig, w_orig), dtype=bool)
                m[by:by + bh, bx:bx + bw] = True
                candidates.append({
                    "mask": m, "bbox": ow_box, "area": bw * bh,
                    "cls_id": v_info[0], "raw_name": raw_nm,
                    "cls_name": v_info[1], "category": v_info[2],
                    "conf": float(c_score), "source": "open_vocab"
                })
        else:
            # Explicit custom vocabulary path
            world_model = get_yolo_world_model(self.model_path)
            if world_model is not None:
                try:
                    if not self._model_classes_set:
                        _set_world_classes_cached(world_model, self.active_classes)
                        self._model_classes_set = True
                    preds = world_model.predict(image_pred, imgsz=self.imgsz, conf=self.conf, iou=self.iou, agnostic_nms=True, verbose=False)
                    if preds and len(preds) > 0 and len(preds[0].boxes) > 0:
                        boxes = preds[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        cls_indices = boxes.cls.cpu().numpy().astype(int)
                        conf_scores = boxes.conf.cpu().numpy()
                        for i in range(min(16, len(cls_indices))):
                            x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                            bx = max(0, min(w_orig - 1, x1))
                            by = max(0, min(h_orig - 1, y1))
                            bw = max(1, min(w_orig - bx, x2 - x1))
                            bh = max(1, min(h_orig - by, y2 - y1))
                            raw_nm = self.active_classes[cls_indices[i]] if cls_indices[i] < len(self.active_classes) else "object"
                            if raw_nm == "bowl" and is_deep_space:
                                raw_nm = "planet earth"
                            v_info = OPEN_VOCAB_MAP.get(raw_nm.lower(), ("0899", raw_nm.title(), "General"))
                            m = np.zeros((h_orig, w_orig), dtype=bool)
                            m[by:by + bh, bx:bx + bw] = True
                            candidates.append({
                                "mask": m, "bbox": (bx, by, bw, bh), "area": bw * bh,
                                "cls_id": v_info[0], "raw_name": raw_nm,
                                "cls_name": v_info[1], "category": v_info[2],
                                "conf": float(conf_scores[i]), "source": "custom_world"
                            })
                except Exception as exc:
                    logger.warning("YOLO11n-seg hybrid inference failed (%s).", exc)

        if candidates:
            # Layer back-to-front (largest area first, smaller foreground objects painted on top)
            candidates.sort(key=lambda d: d["area"], reverse=True)
            candidates = candidates[:16]

            label_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
            color_lut = np.zeros((256, c), dtype=np.uint8)
            for idx, cand in enumerate(candidates, 1):
                cand["label_id"] = idx
                label_map[cand["mask"]] = idx

            results: List[RegionDescriptor] = []
            for cand in candidates:
                label_id = cand["label_id"]
                vis_mask = (label_map == label_id)
                if int(np.count_nonzero(vis_mask)) < 16:
                    vis_mask = cand["mask"]

                bx, by, bw, bh = cand["bbox"]
                area = int(np.count_nonzero(cand["mask"]))
                cov_pct = round((area / total_pixels) * 100.0, 2)
                pts = image[vis_mask]
                dom_col = np.median(pts, axis=0).astype(int).tolist()[:c] if pts.size > 0 else [128] * c
                color_lut[label_id] = dom_col

                emb = RegionEmbeddingGenerator.compute(image, (bx, by, bw, bh), vis_mask)
                cls_id = cand["cls_id"]
                category = cand["category"] or "General"
                is_geo_bg = cls_id in ("0210", "0211", "0212", "0213", "0220")
                is_bg = is_geo_bg or cls_id.startswith("09") or category == "Background"
                is_struct = cls_id.startswith("01")
                kind_val = "BG" if is_bg else ("STRUCT" if is_struct else "OBJ")
                prom_val = "BACKGROUND" if is_bg else ("PRIMARY" if cov_pct > 10.0 else ("SECONDARY" if cov_pct >= 1.5 else "MINOR_DETAIL"))

                results.append(RegionDescriptor(
                    region_id=f"{label_id:03d}",
                    class_id=cls_id,
                    class_name=cand["cls_name"],
                    category=category,
                    bbox=(bx, by, bw, bh),
                    dominant_color=dom_col,
                    label_id=label_id,
                    area=area,
                    coverage_pct=cov_pct,
                    kind=kind_val,
                    conf=cand["conf"],
                    prominence=prom_val,
                    role=prom_val,
                    embedding=emb
                ))

            # Fill background with honest [BG] zone
            bg_mask = (label_map == 0)
            if np.any(bg_mask):
                bg_label = len(results) + 1
                label_map[bg_mask] = bg_label
                bg_col = np.median(image[bg_mask], axis=0).astype(int).tolist()[:c]
                color_lut[bg_label] = bg_col
                bg_cls = "0206" if is_deep_space else ("0901" if np.mean(bg_col[:3]) < 45 else "0900")
                bg_name = "Outer Space / Deep Space (Celestial)" if is_deep_space else "Ambient Background"
                bg_emb = RegionEmbeddingGenerator.compute(image, (0, 0, w_orig, h_orig), bg_mask)

                results.append(RegionDescriptor(
                    region_id=f"{bg_label:03d}",
                    class_id=bg_cls,
                    class_name=bg_name,
                    category="Background",
                    bbox=(0, 0, w_orig, h_orig),
                    dominant_color=bg_col,
                    label_id=bg_label,
                    area=int(np.count_nonzero(bg_mask)),
                    coverage_pct=round(float(np.mean(bg_mask) * 100.0), 2),
                    kind="BG",
                    conf=1.00,
                    prominence="BACKGROUND",
                    role="BACKGROUND",
                    embedding=bg_emb
                ))

            texts = _collect_ocr()
            self.last_texts = texts
            return results, label_map, color_lut, texts

        # 4. Fallback chain (abstract / solid / non-object images -> honest BG)
        regs, l_map, c_lut, fb_texts = self.fallback_extractor.extract(image)
        texts = _collect_ocr(fallback=fb_texts)
        self.last_texts = texts
        # Compute embeddings for fallback regions
        for r in regs:
            if r.embedding is None:
                r.embedding = RegionEmbeddingGenerator.compute(image, r.bbox)
        return regs, l_map, c_lut, texts

