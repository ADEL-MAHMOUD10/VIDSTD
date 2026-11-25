"""
VIDSTD Human Pose & Action Kinematics Classifier.
Analyzes bounding-box dynamics, vertical oscillations, multi-person collisions,
and posture transitions to automatically classify human actions without rigid templates:
  - ACTION:Workout/Repetitive_Exercise (Reps_Counted: N)
  - ACTION:Physical_Altercation/Fight_or_Fast_Contact
  - ACTION:Falling/Crouching
  - ACTION:Running / Walking / Stationary
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class HumanActionKinematics:
    """
    Classifies human actions from temporal tracking trajectory and pose dynamics.
    """

    @classmethod
    def classify_action(
        cls,
        track: Any,
        all_tracks: Optional[List[Any]] = None,
        fps: float = 24.0
    ) -> str:
        """
        Classifies human action for a given person track based on motion history and multi-person kinematics.
        """
        # Only classify person entities (class_id == "0001" or "Person" in class_name)
        cls_name = getattr(track, "class_name", "")
        cls_id = getattr(track, "class_id", "")
        if cls_id != "0001" and "person" not in cls_name.lower():
            return ""

        history = getattr(track, "motion_history", [])
        if not history or len(history) < 3:
            return "ACTION:Stationary"

        # Extract centroids, heights, and widths over time
        frames = [h[0] for h in history]
        boxes = [h[1] for h in history]
        
        cys = np.array([b[1] + b[3] / 2.0 for b in boxes], dtype=float)
        cxs = np.array([b[0] + b[2] / 2.0 for b in boxes], dtype=float)
        heights = np.array([b[3] for b in boxes], dtype=float)
        widths = np.array([b[2] for b in boxes], dtype=float)

        # -------------------------------------------------------------
        # 1. Multi-Person Collision & Fast Contact (Fight / Altercation)
        # -------------------------------------------------------------
        if all_tracks:
            my_trk_id = getattr(track, "track_id", "")
            for other in all_tracks:
                other_id = getattr(other, "track_id", "")
                other_cls = getattr(other, "class_id", "")
                if other_id == my_trk_id or (other_cls != "0001" and "person" not in getattr(other, "class_name", "").lower()):
                    continue
                
                other_history = getattr(other, "motion_history", [])
                if not other_history or len(other_history) < 2:
                    continue

                # Align frames and compute distance & IoU
                other_frame_map = {h[0]: h[1] for h in other_history}
                common_frames = [f for f in frames if f in other_frame_map]
                if len(common_frames) >= 2:
                    high_vel_collisions = 0
                    for cf in common_frames:
                        my_b = boxes[frames.index(cf)]
                        oth_b = other_frame_map[cf]
                        # Compute IoU
                        iou = cls._compute_iou(my_b, oth_b)
                        # Center distance
                        my_c = (my_b[0] + my_b[2]/2.0, my_b[1] + my_b[3]/2.0)
                        oth_c = (oth_b[0] + oth_b[2]/2.0, oth_b[1] + oth_b[3]/2.0)
                        dist = ((my_c[0] - oth_c[0])**2 + (my_c[1] - oth_c[1])**2)**0.5
                        
                        # High velocity / sudden proximity
                        if iou > 0.10 or dist < max(my_b[2], oth_b[2]) * 0.95:
                            high_vel_collisions += 1

                    if high_vel_collisions >= 2:
                        common_dframes = np.maximum(1.0, np.diff(np.array(common_frames, dtype=float)))
                        oth_cxs = np.array([other_frame_map[cf][0] + other_frame_map[cf][2]/2.0 for cf in common_frames])
                        my_common_cxs = np.array([boxes[frames.index(cf)][0] + boxes[frames.index(cf)][2]/2.0 for cf in common_frames])
                        rel_diff = np.abs(np.diff(my_common_cxs - oth_cxs)) / common_dframes
                        max_rel_speed = float(np.max(rel_diff)) if len(rel_diff) > 0 else 0.0

                        dframes_all = np.maximum(1.0, np.diff(np.array(frames, dtype=float)))
                        dx_diff = np.abs(np.diff(cxs))
                        dy_diff = np.abs(np.diff(cys))
                        frame_speeds = (dx_diff + dy_diff) / dframes_all
                        max_speed = float(np.max(frame_speeds)) if len(frame_speeds) > 0 else 0.0
                        if max_speed >= 2.5 or max_rel_speed >= 4.0:
                            return "ACTION:Physical_Altercation/Fight_or_Fast_Contact"

        # -------------------------------------------------------------
        # 2. Workout / Repetitive Vertical Oscillation
        # -------------------------------------------------------------
        reps = cls._count_vertical_oscillations(cys, min_amp=5.0)
        if reps >= 2:
            return f"ACTION:Workout/Repetitive_Exercise (Reps_Counted: {reps})"

        # -------------------------------------------------------------
        # 3. Rapid Downward Drop (Falling / Crouching)
        # -------------------------------------------------------------
        if len(cys) >= 3:
            dy_drop = cys[-1] - cys[0]
            frame_span = max(1, frames[-1] - frames[0])
            aspect_initial = widths[0] / max(1.0, heights[0])
            aspect_final = widths[-1] / max(1.0, heights[-1])
            # Rapid downward drop (normalized by actual frame span) or flattening posture
            if (dy_drop > 25.0 and frame_span <= 12) or (aspect_initial < 0.55 and aspect_final > 0.90):
                return "ACTION:Falling/Crouching"

        # -------------------------------------------------------------
        # 4. Normal Locomotion (Running, Walking, Stationary) - normalized per source frame (sample-rate invariant)
        # -------------------------------------------------------------
        if len(cxs) >= 2:
            dframes = np.maximum(1.0, np.diff(np.array(frames, dtype=float)))
            dxs = np.abs(np.diff(cxs))
            dys = np.abs(np.diff(cys))
            # Scale normalized per-frame speed to 24 FPS baseline so sparse sample_fps does not inflate speed
            fps_norm = max(1.0, fps) / 24.0
            mean_speed = float(np.mean((dxs + dys) / dframes)) * fps_norm
            if mean_speed > 10.0:
                return "ACTION:Running"
            elif mean_speed > 2.0:
                return "ACTION:Walking"

        return "ACTION:Stationary"

    @staticmethod
    def _count_vertical_oscillations(values: np.ndarray, min_amp: float = 5.0) -> int:
        """Counts periodic oscillations (reps) with minimum amplitude."""
        if len(values) < 5:
            return 0

        # Find local peaks and valleys
        diffs = np.diff(values)
        if len(diffs) < 2:
            return 0

        sign_changes = np.diff(np.sign(diffs))
        peaks = np.where(sign_changes < 0)[0] + 1
        valleys = np.where(sign_changes > 0)[0] + 1

        if len(peaks) == 0 or len(valleys) == 0:
            return 0

        # For each peak, verify that an adjacent valley differs by at least min_amp
        valid_reps = sum(1 for p in peaks if any(abs(values[p] - values[v]) >= min_amp for v in valleys))
        return valid_reps

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
