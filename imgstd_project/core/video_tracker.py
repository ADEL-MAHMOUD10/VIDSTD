"""
VIDSTD Temporal Tracker & Scene Analyzer Module — V01.
Provides ultra-fast scene-cut detection (< 0.2 ms), temporal multi-object tracking,
and [TIMELINE_INDEX] compilation for zero-vision AI video comprehension.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from .action_kinematics import HumanActionKinematics


@dataclass
class TrackedEntity:
    track_id: str                      # e.g. "TRK:001"
    kind: str                          # "OBJ" or "STRUCT"
    class_id: str                      # e.g. "0001", "0003", "0102"
    class_name: str                    # e.g. "Person (Human)"
    conf: float
    first_frame: int
    last_frame: int
    first_time_sec: float
    last_time_sec: float
    start_box: Tuple[int, int, int, int]    # (x, y, w, h)
    end_box: Tuple[int, int, int, int]      # (x, y, w, h)
    current_box: Tuple[int, int, int, int]  # (x, y, w, h)
    dominant_color: List[int]
    role: str                          # "PRIMARY", "SECONDARY", "MINOR_DETAIL"
    start_cov_pct: float = 0.0
    end_cov_pct: float = 0.0
    is_reflection: bool = False
    likely_reflection_or_inset: bool = False
    contained_in: Optional[str] = None
    parent_structure: Optional[str] = None
    action: Optional[str] = None
    canvas_width: int = 1920
    closed: bool = False
    motion_history: List[Tuple[int, Tuple[int, int, int, int]]] = field(default_factory=list)

    @property
    def motion_summary(self) -> str:
        """Computes semantic trajectory: Zoom-In/Expanding, Zoom-Out/Shrinking, Static, Pan-Left, etc."""
        sx, sy, sw, sh = self.start_box
        ex, ey, ew, eh = self.end_box
        start_area = max(1, sw * sh)
        end_area = max(1, ew * eh)
        ratio = end_area / start_area
        dx = ex - sx
        dy = ey - sy
        dist = (dx**2 + dy**2)**0.5

        if ratio >= 1.25:
            return "Zoom-In/Expanding"
        elif ratio <= 0.75:
            return "Zoom-Out/Shrinking"
        elif dist < 8:
            return f"Static({sx},{sy},{sw},{sh})"
        elif abs(dx) > abs(dy):
            return "Pan-Right/Moving" if dx > 0 else "Pan-Left/Moving"
        else:
            return "Moving-Down" if dy > 0 else "Moving-Up"

    @property
    def path_summary(self) -> str:
        """Computes semantic trajectory description: Static, Left->Right, Moving, etc."""
        sx, sy, sw, sh = self.start_box
        ex, ey, ew, eh = self.end_box
        dx = ex - sx
        dy = ey - sy
        dist = (dx**2 + dy**2)**0.5

        if dist < 8:
            return f"Static({sx},{sy},{sw},{sh})"

        # Describe horizontal movement
        h_dir = "Left" if sx < ex else "Right"
        v_dir = "Down" if sy < ey else "Up"
        
        # Determine proportional canvas position zones (independent of resolution)
        cw = max(1, int(getattr(self, "canvas_width", 1920)))
        left_bound = cw * 0.33
        right_bound = cw * 0.67
        start_zone = "Left" if sx < left_bound else ("Right" if sx > right_bound else "Center")
        end_zone = "Left" if ex < left_bound else ("Right" if ex > right_bound else "Center")

        return f"{start_zone}({sx},{sy})->{end_zone}({ex},{ey})"


@dataclass
class TrackedText:
    text_track_id: str                 # e.g. "TXT_TRK:001"
    text: str
    conf: float
    first_frame: int
    last_frame: int
    first_time_sec: float
    last_time_sec: float
    box: Tuple[int, int, int, int]     # (x, y, w, h)
    closed: bool = False


@dataclass
class SceneRecord:
    scene_id: str                      # e.g. "SCENE:001"
    scene_num: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    context: str = "General scene"


class SceneCutDetector:
    """
    Ultra-fast downscaled MAD scene cut & motion detector (< 0.2 ms per frame).
    Downscales to 64x64 via INTER_NEAREST and evaluates Mean Absolute Difference.
    """

    def __init__(self, cut_threshold: float = 38.0, min_scene_len: int = 0):
        self.cut_threshold = cut_threshold
        self.min_scene_len = min_scene_len
        self._frames_since_last_cut = 0

    @staticmethod
    def _hist_correlation(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
        """Computes fast 16-bin color histogram correlation on 64x64 thumbnails (< 0.05 ms)."""
        channels = [0, 1] if a_u8.ndim == 3 and a_u8.shape[2] >= 2 else [0]
        ranges = [0, 256, 0, 256] if len(channels) == 2 else [0, 256]
        bins = [16, 16] if len(channels) == 2 else [16]
        ha = cv2.calcHist([a_u8], channels, None, bins, ranges)
        hb = cv2.calcHist([b_u8], channels, None, bins, ranges)
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))

    def evaluate(self, curr_frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> Tuple[str, float]:
        """
        Evaluates frame change against preceding frame.
        
        Returns:
            frame_type: 'I' (first frame or scene cut), 'S' (skip/identical), or 'P' (predicted delta)
            mad: Mean Absolute Difference score
        """
        if prev_frame is None:
            self._frames_since_last_cut = 0
            return "I", 0.0

        self._frames_since_last_cut += 1

        # Exact pixel match check (< 0.15 ms)
        if np.array_equal(curr_frame, prev_frame):
            return "S", 0.0

        # Downscaled 64x64 MAD evaluation
        curr_u8 = cv2.resize(curr_frame, (64, 64), interpolation=cv2.INTER_NEAREST)
        prev_u8 = cv2.resize(prev_frame, (64, 64), interpolation=cv2.INTER_NEAREST)
        mad = float(np.mean(np.abs(curr_u8.astype(np.int16) - prev_u8.astype(np.int16))))

        if mad >= self.cut_threshold and self._frames_since_last_cut >= self.min_scene_len:
            if self.min_scene_len == 0 or mad >= 65.0 or self._hist_correlation(curr_u8, prev_u8) < 0.78:
                self._frames_since_last_cut = 0
                return "CUT", mad

        return "P", mad


class TemporalObjectTracker:
    """
    Multi-Object Temporal Tracker & [TIMELINE_INDEX] Compiler.
    Maintains persistent entity tracks across I-frames and propagates trajectories across P-frames.
    """

    def __init__(self, fps: float = 24.0, iou_thresh: float = 0.30):
        self.fps = max(1.0, fps)
        self.iou_thresh = iou_thresh
        self.tracks: List[TrackedEntity] = []
        self.text_tracks: List[TrackedText] = []
        self.scenes: List[SceneRecord] = []
        self.camera_events: List[Dict[str, Any]] = []
        self._next_trk_id = 1
        self._next_txt_id = 1
        self._current_scene_idx = 1
        self._scene_start_f = 0
        self._current_scene_context = "General visual environment"
        self._last_seen_frame: Optional[int] = None

    def set_camera_events(self, events: List[Dict[str, Any]]):
        """Attaches camera ego-motion and seismic events."""
        self.camera_events = events

    def start_new_scene(self, start_frame: int, context: str = "General scene"):
        """Closes previous scene (and its active entity/text tracks to prevent cross-scene bleeding) and opens a new scene."""
        current_time = start_frame / self.fps
        if self.scenes:
            prev = self.scenes[-1]
            prev.end_frame = max(prev.start_frame, start_frame - 1)
            prev.end_time_sec = round(prev.end_frame / self.fps, 2)

        for trk in self.tracks:
            trk.closed = True
        for tt in self.text_tracks:
            tt.closed = True

        self._scene_start_f = start_frame
        self._current_scene_context = context
        new_scene = SceneRecord(
            scene_id=f"SCENE:{self._current_scene_idx:03d}",
            scene_num=self._current_scene_idx,
            start_frame=start_frame,
            end_frame=start_frame,
            start_time_sec=round(current_time, 2),
            end_time_sec=round(current_time, 2),
            context=context
        )
        self.scenes.append(new_scene)
        self._current_scene_idx += 1

    def update_scene_context(self, context: str):
        """Updates the semantic context of the active scene."""
        self._current_scene_context = context
        if self.scenes:
            self.scenes[-1].context = context

    def register_keyframe(
        self,
        frame_idx: int,
        regions: List[Any],
        texts: List[Any],
        scene_summary: str = "",
        frame_width: Optional[int] = None
    ):
        """
        Processes an I-Frame with full neural vision detections.
        Matches detected objects and texts to active tracks in the current scene or creates new ones.
        """
        curr_time = round(frame_idx / self.fps, 2)
        if frame_width is not None:
            self._canvas_width = max(1, int(frame_width))
        canvas_w = getattr(self, "_canvas_width", 1920)
        if scene_summary:
            self.update_scene_context(scene_summary)

        # 1. Update text tracks (scoped to active scene)
        for t in texts:
            t_txt = getattr(t, "text", "") if hasattr(t, "text") else t.get("text", "")
            t_conf = float(getattr(t, "confidence", 1.0) if hasattr(t, "confidence") else t.get("confidence", 1.0))
            t_bbox = getattr(t, "bbox", (0, 0, 0, 0)) if hasattr(t, "bbox") else tuple(t.get("box", [0, 0, 0, 0]))

            matched = False
            for tt in self.text_tracks:
                if not tt.closed and tt.text == t_txt:
                    tt.last_frame = frame_idx
                    tt.last_time_sec = curr_time
                    tt.box = t_bbox
                    matched = True
                    break

            if not matched and t_txt.strip():
                new_tt = TrackedText(
                    text_track_id=f"TXT_TRK:{self._next_txt_id:03d}",
                    text=t_txt.strip(),
                    conf=t_conf,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    first_time_sec=curr_time,
                    last_time_sec=curr_time,
                    box=t_bbox
                )
                self.text_tracks.append(new_tt)
                self._next_txt_id += 1

        # 2. Update object and structural tracks
        for r in regions:
            kind = getattr(r, "kind", "OBJ") if hasattr(r, "kind") else r.get("kind", "OBJ")
            if kind == "BG":
                continue

            cls_id = getattr(r, "class_id", "0000") if hasattr(r, "class_id") else r.get("class_id", "0000")
            cls_name = getattr(r, "class_name", "Unknown") if hasattr(r, "class_name") else r.get("class_name", "Unknown")
            conf = float(getattr(r, "conf", 1.0) if hasattr(r, "conf") else r.get("confidence", 1.0))
            bbox = getattr(r, "bbox", (0, 0, 0, 0)) if hasattr(r, "bbox") else tuple(
                [r["bounding_box"]["x"], r["bounding_box"]["y"], r["bounding_box"]["width"], r["bounding_box"]["height"]]
                if "bounding_box" in r and isinstance(r["bounding_box"], dict)
                else r.get("bbox", [0, 0, 0, 0])
            )
            col = getattr(r, "dominant_color", [0, 0, 0]) if hasattr(r, "dominant_color") else r.get("dominant_color_rgb", [0, 0, 0])
            role = getattr(r, "role", "SECONDARY") if hasattr(r, "role") else r.get("role", "SECONDARY")
            is_refl = getattr(r, "is_reflection", False) if hasattr(r, "is_reflection") else r.get("is_reflection", False)
            contained_in = getattr(r, "contained_in", None) if hasattr(r, "contained_in") else r.get("contained_in")
            parent_struct = getattr(r, "parent_structure", None) if hasattr(r, "parent_structure") else r.get("parent_structure")

            # Find matching active track
            best_track = None
            best_score = -1.0

            if hasattr(r, "coverage_pct"):
                cov_val = float(r.coverage_pct)
            elif isinstance(r, dict) and "coverage_percentage" in r:
                cov_val = float(r["coverage_percentage"])
            elif isinstance(r, dict) and "coverage_pct" in r:
                cov_val = float(r["coverage_pct"])
            else:
                cov_val = 0.0
            for trk in self.tracks:
                if trk.closed or trk.class_id != cls_id or trk.class_name != cls_name:
                    continue
                # Calculate IoU
                iou_val = self._compute_iou(trk.current_box, bbox)
                # Color distance
                col_dist = np.linalg.norm(np.array(trk.dominant_color) - np.array(col))
                score = iou_val * 0.8 + max(0, 1.0 - col_dist / 100.0) * 0.2
                if score > best_score and (iou_val >= self.iou_thresh or score > 0.45):
                    best_score = score
                    best_track = trk

            if best_track:
                best_track.last_frame = frame_idx
                best_track.last_time_sec = curr_time
                best_track.end_box = bbox
                best_track.current_box = bbox
                best_track.end_cov_pct = cov_val
                best_track.motion_history.append((frame_idx, bbox))
                if is_refl:
                    best_track.is_reflection = True
                    best_track.contained_in = contained_in
            else:
                new_trk = TrackedEntity(
                    track_id=f"TRK:{self._next_trk_id:03d}",
                    kind=kind,
                    class_id=cls_id,
                    class_name=cls_name,
                    conf=conf,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    first_time_sec=curr_time,
                    last_time_sec=curr_time,
                    start_box=bbox,
                    end_box=bbox,
                    current_box=bbox,
                    dominant_color=col,
                    role=role,
                    start_cov_pct=cov_val,
                    end_cov_pct=cov_val,
                    is_reflection=is_refl,
                    likely_reflection_or_inset=is_refl,
                    contained_in=contained_in,
                    parent_structure=parent_struct,
                    canvas_width=canvas_w,
                    motion_history=[(frame_idx, bbox)]
                )
                self.tracks.append(new_trk)
                self._next_trk_id += 1

        self._last_seen_frame = frame_idx

    def propagate_pframe(
        self,
        frame_idx: int,
        curr_frame: np.ndarray,
        prev_frame: np.ndarray,
        mad: float
    ) -> str:
        """
        Propagates active object bounding boxes on intermediate P-Frames without running neural models.
        Uses fast template matching / centroid optical flow displacement.
        
        Returns:
            sem_delta string: e.g. "TRK:001->BOX:128,500,90,260"
        """
        curr_time = round(frame_idx / self.fps, 2)
        prev_expected = self._last_seen_frame if self._last_seen_frame is not None else (frame_idx - 1)
        deltas = []

        # If minimal motion (MAD <= 1.0), all tracks remain stationary
        if mad <= 1.0:
            for trk in self.tracks:
                if not trk.closed and trk.last_frame == prev_expected:
                    trk.last_frame = frame_idx
                    trk.last_time_sec = curr_time
            # Also extend active text tracks
            for tt in self.text_tracks:
                if not tt.closed and tt.last_frame == prev_expected:
                    tt.last_frame = frame_idx
                    tt.last_time_sec = curr_time
            self._last_seen_frame = frame_idx
            return ""

        h, w = curr_frame.shape[:2]

        for trk in self.tracks:
            # Only propagate active, moving OBJ tracks (not closed or static structures)
            if trk.closed:
                continue
            if trk.kind == "STRUCT" or trk.last_frame != prev_expected:
                if trk.last_frame == prev_expected:
                    trk.last_frame = frame_idx
                    trk.last_time_sec = curr_time
                continue

            bx, by, bw, bh = trk.current_box
            if bw < 8 or bh < 8:
                trk.last_frame = frame_idx
                trk.last_time_sec = curr_time
                continue

            # Crop template from prev_frame with safety bounds
            y1, y2 = max(0, by), min(h, by + bh)
            x1, x2 = max(0, bx), min(w, bx + bw)
            tmpl = prev_frame[y1:y2, x1:x2]

            # Search in a local window (+/- 20 px) in curr_frame
            win_pad = 20
            sy1, sy2 = max(0, y1 - win_pad), min(h, y2 + win_pad)
            sx1, sx2 = max(0, x1 - win_pad), min(w, x2 + win_pad)
            search_area = curr_frame[sy1:sy2, sx1:sx2]

            if search_area.shape[0] >= tmpl.shape[0] and search_area.shape[1] >= tmpl.shape[1]:
                # Fast template match
                res = cv2.matchTemplate(search_area, tmpl, cv2.TM_SQDIFF_NORMED)
                min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                if min_val < 0.25:
                    new_x = sx1 + min_loc[0]
                    new_y = sy1 + min_loc[1]
                    new_box = (new_x, new_y, bw, bh)
                    trk.current_box = new_box
                    trk.end_box = new_box
                    trk.last_frame = frame_idx
                    trk.last_time_sec = curr_time
                    trk.motion_history.append((frame_idx, new_box))
                    deltas.append(f"{trk.track_id}->BOX:{new_x},{new_y},{bw},{bh}")
                    continue

            # Fallback: maintain position
            trk.last_frame = frame_idx
            trk.last_time_sec = curr_time

        # Extend active text tracks
        for tt in self.text_tracks:
            if not tt.closed and tt.last_frame == prev_expected:
                tt.last_frame = frame_idx
                tt.last_time_sec = curr_time

        self._last_seen_frame = frame_idx
        return "|".join(deltas)

    def finalize(self, total_frames: int):
        """Finalizes the scene and track ranges."""
        if self.scenes:
            self.scenes[-1].end_frame = max(self.scenes[-1].start_frame, total_frames - 1)
            self.scenes[-1].end_time_sec = round(self.scenes[-1].end_frame / self.fps, 2)

    def compile_timeline_index(self) -> str:
        """
        Compiles the human-and-AI-readable [TIMELINE_INDEX] container block.
        """
        lines = ["[TIMELINE_INDEX]"]

        # 1. Scenes
        for sc in self.scenes:
            ctx_esc = sc.context.replace('\\', '\\\\').replace('"', '\\"').replace('|', '\\|').replace('\n', ' ')
            lines.append(
                f"SCENE:{sc.scene_num:03d}|FRAMES:{sc.start_frame}-{sc.end_frame}|"
                f"TIME:{sc.start_time_sec:.2f}s-{sc.end_time_sec:.2f}s|CONTEXT:\"{ctx_esc}\""
            )

        # 1b. Camera Ego-Motion & Seismic Events
        for ce in self.camera_events:
            lines.append(
                f"CAM_MOTION:{ce.get('motion_id', 'CAM:001')}|TYPE:{ce.get('type', 'Static')}|"
                f"TIME:{ce.get('start_time_sec', 0.0):.2f}s->{ce.get('end_time_sec', 0.0):.2f}s|"
                f"FRAMES:{ce.get('start_frame', 0)}->{ce.get('end_frame', 0)}"
            )

        # 2. Tracked Objects & Structures (Event-Deduplicated Lifecycle Format)
        for trk in self.tracks:
            if not trk.action:
                act = HumanActionKinematics.classify_action(trk, self.tracks, fps=self.fps)
                if act:
                    trk.action = act
            action_tag = f"|{trk.action}" if trk.action else ""
            path_str = trk.path_summary
            motion_str = trk.motion_summary
            rgb_str = ",".join(str(v) for v in trk.dominant_color[:3])
            inside_str = f"|INSIDE_OF:{trk.contained_in}" if trk.is_reflection and trk.contained_in else ""
            cov_str = f"{trk.start_cov_pct:.1f}%->{trk.end_cov_pct:.1f}%"
            lines.append(
                f"{trk.track_id}|KIND:{trk.kind}|CLS:{trk.class_id}({trk.class_name})|"
                f"CONF:{trk.conf:.2f}|TIME:{trk.first_time_sec:.2f}s->{trk.last_time_sec:.2f}s|"
                f"FRAMES:{trk.first_frame}->{trk.last_frame}|FIRST_F:{trk.first_frame}({trk.first_time_sec:.2f}s)|"
                f"LAST_F:{trk.last_frame}({trk.last_time_sec:.2f}s)|MOTION:{motion_str}|PATH:{path_str}|"
                f"COV:{cov_str}|RGB:{rgb_str}|ROLE:{trk.role}{inside_str}{action_tag}"
            )

        # 3. Text Tracks (Event-Deduplicated Lifecycle Format)
        for tt in self.text_tracks:
            bx, by, bw, bh = tt.box
            esc_txt = tt.text.replace('\\', '\\\\').replace('"', '\\"').replace('|', '\\|').replace('\n', '\\n')
            lines.append(
                f"{tt.text_track_id}|STR:\"{esc_txt}\"|CONF:{tt.conf:.2f}|"
                f"TIME:{tt.first_time_sec:.2f}s->{tt.last_time_sec:.2f}s|"
                f"FRAMES:{tt.first_frame}->{tt.last_frame}|"
                f"FIRST_F:{tt.first_frame}({tt.first_time_sec:.2f}s)|"
                f"LAST_F:{tt.last_frame}({tt.last_time_sec:.2f}s)|BOX:{bx},{by},{bw},{bh}"
            )

        lines.append("[/TIMELINE_INDEX]")
        return "\n".join(lines)

    def to_json_summary(self) -> Dict[str, Any]:
        """Returns structured JSON summary for instant Non-Vision LLM consumption."""
        return {
            "scene_count": len(self.scenes),
            "scenes": [
                {
                    "scene_id": sc.scene_id,
                    "start_frame": sc.start_frame,
                    "end_frame": sc.end_frame,
                    "duration_sec": round(sc.end_time_sec - sc.start_time_sec, 2),
                    "context": sc.context
                }
                for sc in self.scenes
            ],
            "tracked_objects_count": len(self.tracks),
            "tracked_objects": [
                {
                    "track_id": trk.track_id,
                    "kind": trk.kind,
                    "class_id": trk.class_id,
                    "class_name": trk.class_name,
                    "confidence": trk.conf,
                    "start_time_sec": trk.first_time_sec,
                    "end_time_sec": trk.last_time_sec,
                    "start_box": {"x": trk.start_box[0], "y": trk.start_box[1], "w": trk.start_box[2], "h": trk.start_box[3]},
                    "end_box": {"x": trk.end_box[0], "y": trk.end_box[1], "w": trk.end_box[2], "h": trk.end_box[3]},
                    "path_summary": trk.path_summary,
                    "role": trk.role,
                    "is_reflection": trk.is_reflection,
                    "contained_in": trk.contained_in,
                }
                for trk in self.tracks
            ],
            "text_events_count": len(self.text_tracks),
            "text_events": [
                {
                    "text_track_id": tt.text_track_id,
                    "text": tt.text,
                    "confidence": tt.conf,
                    "start_time_sec": tt.first_time_sec,
                    "end_time_sec": tt.last_time_sec,
                    "box": {"x": tt.box[0], "y": tt.box[1], "w": tt.box[2], "h": tt.box[3]}
                }
                for tt in self.text_tracks
            ]
        }

    @staticmethod
    def _compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])

        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        union = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter
        if union <= 0:
            return 0.0
        return inter / union
