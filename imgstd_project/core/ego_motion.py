"""
VIDSTD Global Camera Ego-Motion & Seismic Sensor Module.
Operates on downscaled 64x64 frames in < 0.5 ms using optical flow
and affine motion decomposition to classify:
  - Violent Tremor / Shake (Earthquake / Impact Signature)
  - Forward Driving / Zoom In (Radial Expansion)
  - Reversing / Zoom Out (Radial Contraction)
  - Panning / Turning Left or Right
  - Static Tripod Camera
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


class CameraEgoMotionTracker:
    """
    Ultra-fast Global Camera Ego-Motion & Seismic Detector (< 0.5 ms per frame).
    Computes global translational variance, directional sign reversals,
    and center-radial divergence to classify ego-motion without external sensors.
    """

    def __init__(self, fps: float = 24.0, downsample_size: Tuple[int, int] = (64, 64)):
        self.fps = max(1.0, fps)
        self.dw, self.dh = downsample_size
        self._prev_small_gray: Optional[np.ndarray] = None
        self._history: List[Dict[str, Any]] = []

        # Radial unit grid for 64x64
        ys, xs = np.mgrid[0:self.dh, 0:self.dw]
        cx, cy = (self.dw - 1) / 2.0, (self.dh - 1) / 2.0
        rx = xs - cx
        ry = ys - cy
        dist = np.sqrt(rx**2 + ry**2)
        dist[dist == 0] = 1.0
        self._unit_rx = rx / dist
        self._unit_ry = ry / dist

    def process_frame(
        self,
        frame_idx: int,
        curr_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes single frame in < 0.4 ms, calculating translation and radial flow.
        """
        curr_time = round(frame_idx / self.fps, 2)

        # Downscale to 64x64 grayscale
        if curr_frame.ndim == 3:
            curr_small = cv2.resize(curr_frame, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST)
            curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_RGB2GRAY)
        else:
            curr_gray = cv2.resize(curr_frame, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST)

        if prev_frame is None and self._prev_small_gray is None:
            self._prev_small_gray = curr_gray
            record = {
                "frame_idx": frame_idx,
                "time_sec": curr_time,
                "dx": 0.0,
                "dy": 0.0,
                "radial_div": 0.0,
                "motion_type": "Static"
            }
            self._history.append(record)
            return record

        prev_gray = self._prev_small_gray
        self._prev_small_gray = curr_gray

        # Optical flow on 64x64 (ultra-fast ~0.15ms)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=1, winsize=11,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )

        u = flow[:, :, 0]
        v = flow[:, :, 1]

        mean_u = float(np.mean(u))
        mean_v = float(np.mean(v))

        # Radial divergence: positive = expansion (forward driving/zoom-in)
        radial_div = float(np.mean(u * self._unit_rx + v * self._unit_ry))

        mag = float(np.sqrt(mean_u**2 + mean_v**2))

        # Frame-level instant classification
        if radial_div >= 0.35:
            m_type = "Forward_Driving / Zoom_In"
        elif radial_div <= -0.35:
            m_type = "Reversing / Zoom_Out"
        elif abs(mean_u) > 0.5:
            m_type = "Panning_Right / Turning_Right" if mean_u > 0 else "Panning_Left / Turning_Left"
        elif mag < 0.2:
            m_type = "Static"
        else:
            m_type = "Moving"

        record = {
            "frame_idx": frame_idx,
            "time_sec": curr_time,
            "dx": mean_u,
            "dy": mean_v,
            "radial_div": radial_div,
            "motion_type": m_type
        }
        self._history.append(record)
        return record

    def _classify_window(self, hist_slice: List[Dict[str, Any]], sample_step: int) -> Dict[str, Any]:
        """Classifies a temporal window of optical-flow records into a single camera motion label."""
        n = len(hist_slice)
        dx_vals = np.array([r["dx"] for r in hist_slice])
        dy_vals = np.array([r["dy"] for r in hist_slice])
        rad_vals = np.array([r["radial_div"] for r in hist_slice])

        # 1. Seismic / Violent Shake Detection (meaningful only on dense consecutive frames)
        if sample_step == 1 and n >= 4:
            dx_signs = np.sign(dx_vals[1:]) != np.sign(dx_vals[:-1])
            dy_signs = np.sign(dy_vals[1:]) != np.sign(dy_vals[:-1])
            sign_flips = int(np.sum(dx_signs) + np.sum(dy_signs))
            trans_var = float(np.var(dx_vals) + np.var(dy_vals))
            if trans_var > 0.30 and sign_flips >= max(2, n // 4):
                return {
                    "type": "Violent_Tremor/Shake (Earthquake_or_Impact_Signature)",
                    "start_time_sec": hist_slice[0]["time_sec"],
                    "end_time_sec": hist_slice[-1]["time_sec"],
                    "start_frame": hist_slice[0]["frame_idx"],
                    "end_frame": hist_slice[-1]["frame_idx"],
                    "variance": round(trans_var, 3),
                    "oscillations": sign_flips,
                }

        # 2. Dominant radial motion (Driving / Zooming)
        mean_rad = float(np.mean(rad_vals))
        if mean_rad >= 0.25:
            return {
                "type": "Forward_Driving / Zoom_In (Radial_Expansion)",
                "start_time_sec": hist_slice[0]["time_sec"],
                "end_time_sec": hist_slice[-1]["time_sec"],
                "start_frame": hist_slice[0]["frame_idx"],
                "end_frame": hist_slice[-1]["frame_idx"],
                "radial_divergence": round(mean_rad, 3),
            }
        elif mean_rad <= -0.25:
            return {
                "type": "Reversing / Zoom_Out (Radial_Contraction)",
                "start_time_sec": hist_slice[0]["time_sec"],
                "end_time_sec": hist_slice[-1]["time_sec"],
                "start_frame": hist_slice[0]["frame_idx"],
                "end_frame": hist_slice[-1]["frame_idx"],
                "radial_divergence": round(mean_rad, 3),
            }

        # 3. Dominant horizontal panning
        mean_dx = float(np.mean(dx_vals))
        if mean_dx > 0.4:
            return {
                "type": "Panning_Right / Turning_Right",
                "start_time_sec": hist_slice[0]["time_sec"],
                "end_time_sec": hist_slice[-1]["time_sec"],
                "start_frame": hist_slice[0]["frame_idx"],
                "end_frame": hist_slice[-1]["frame_idx"],
            }
        elif mean_dx < -0.4:
            return {
                "type": "Panning_Left / Turning_Left",
                "start_time_sec": hist_slice[0]["time_sec"],
                "end_time_sec": hist_slice[-1]["time_sec"],
                "start_frame": hist_slice[0]["frame_idx"],
                "end_frame": hist_slice[-1]["frame_idx"],
            }

        return {
            "type": "Static",
            "start_time_sec": hist_slice[0]["time_sec"],
            "end_time_sec": hist_slice[-1]["time_sec"],
            "start_frame": hist_slice[0]["frame_idx"],
            "end_frame": hist_slice[-1]["frame_idx"],
        }

    def compile_timeline_events(self, sample_step: int = 1) -> List[Dict[str, Any]]:
        """
        Consolidates frame-by-frame history into segmented camera motion interval events,
        avoiding collapsing an entire multi-minute video into a single averaged state.
        """
        if not self._history:
            return [{
                "motion_id": "CAM:001",
                "type": "Static",
                "start_time_sec": 0.0,
                "end_time_sec": 0.0,
                "start_frame": 0,
                "end_frame": 0
            }]

        total_frames = len(self._history)
        win_len = total_frames if total_frames <= 24 else max(6, min(24, int(round(self.fps * 1.0 / max(1, sample_step)))))

        raw_segments: List[Dict[str, Any]] = []
        idx = 0
        while idx < total_frames:
            end_idx = min(total_frames, idx + win_len)
            if total_frames - end_idx < win_len // 2 and end_idx < total_frames:
                end_idx = total_frames
            raw_segments.append(self._classify_window(self._history[idx:end_idx], sample_step))
            idx = end_idx

        # Merge consecutive segments of identical motion type
        merged: List[Dict[str, Any]] = []
        for seg in raw_segments:
            if merged and merged[-1]["type"] == seg["type"]:
                merged[-1]["end_time_sec"] = seg["end_time_sec"]
                merged[-1]["end_frame"] = seg["end_frame"]
            else:
                merged.append(dict(seg))

        events: List[Dict[str, Any]] = []
        for m_idx, ev in enumerate(merged, 1):
            ev["motion_id"] = f"CAM:{m_idx:03d}"
            events.append(ev)
        return events
