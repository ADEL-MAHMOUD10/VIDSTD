"""
VIDSTD Production-Grade Media I/O Module.
Implements Zero-Copy Fast Frame Ingestion, Hardware Keyframe Hinting,
and Lossless Audio Stream Extraction & Remuxing via PyAV (FFmpeg C-API),
with seamless fallback to OpenCV (cv2).
"""

import os
import io
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

# Try importing PyAV (av)
try:
    import av
    HAS_AV = True
except ImportError:
    av = None
    HAS_AV = False

import cv2


class FastMediaReader:
    """
    High-throughput media reader using PyAV (FFmpeg C-API) multi-threaded decoding
    with automatic fallback to cv2.VideoCapture.
    """

    @classmethod
    def stream_video(
        cls,
        video_input: Union[str, List[np.ndarray], np.ndarray],
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        extract_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Streams video frames one-by-one with O(1) constant memory consumption.
        Never accumulates raw uncompressed video frames into a giant list.
        """
        if isinstance(video_input, (list, np.ndarray)):
            if isinstance(video_input, np.ndarray):
                if video_input.ndim == 4:
                    frames_list = [f for f in video_input]
                else:
                    frames_list = [video_input]
            else:
                frames_list = list(video_input)
            fps = float(target_fps) if target_fps else 24.0
            w = frames_list[0].shape[1] if frames_list else 0
            h = frames_list[0].shape[0] if frames_list else 0
            c = frames_list[0].shape[2] if frames_list and frames_list[0].ndim == 3 else 1

            def in_memory_gen():
                for idx, f in enumerate(frames_list):
                    if max_frames and idx >= max_frames:
                        break
                    yield idx, f, (idx == 0)

            return {
                "fps": fps,
                "width": w,
                "height": h,
                "channels": c,
                "estimated_frames": len(frames_list),
                "has_audio": False,
                "audio_bytes": None,
                "audio_format": None,
                "generator": in_memory_gen,
                "backend": "in_memory"
            }

        if not isinstance(video_input, str) or not os.path.exists(video_input):
            raise FileNotFoundError(f"Video file not found: {video_input}")

        if HAS_AV:
            try:
                return cls._stream_with_pyav(
                    video_input, target_fps=target_fps, max_frames=max_frames, extract_audio=extract_audio
                )
            except Exception:
                pass

        return cls._stream_with_cv2(video_input, target_fps=target_fps, max_frames=max_frames)

    @classmethod
    def read_video(
        cls,
        video_input: Union[str, List[np.ndarray], np.ndarray],
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Reads video frames into RGB uint8 NumPy arrays, extracting hardware keyframe
        hints and lossless audio packets when available.
        """
        stream_info = cls.stream_video(video_input, target_fps=target_fps, max_frames=max_frames)
        frames: List[np.ndarray] = []
        keyframe_hints: List[bool] = []
        for idx, f, is_key in stream_info["generator"]():
            frames.append(f)
            keyframe_hints.append(is_key)

        return {
            "frames": frames,
            "fps": stream_info["fps"],
            "width": stream_info["width"],
            "height": stream_info["height"],
            "channels": stream_info["channels"],
            "keyframe_hints": keyframe_hints,
            "has_audio": stream_info["has_audio"],
            "audio_bytes": stream_info["audio_bytes"],
            "audio_format": stream_info["audio_format"],
            "backend": stream_info["backend"]
        }

    @classmethod
    def _stream_with_pyav(
        cls,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        extract_audio: bool = True
    ) -> Dict[str, Any]:
        container = av.open(video_path)
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")

        v_stream = container.streams.video[0]
        w = v_stream.width
        h = v_stream.height
        c = 3
        fps = float(v_stream.average_rate) if v_stream.average_rate else (target_fps or 24.0)
        est_frames = v_stream.frames if v_stream.frames > 0 else 0
        sample_step = max(1, int(round(fps / target_fps))) if (target_fps and 0 < target_fps < fps) else 1

        has_audio = extract_audio and (len(container.streams.audio) > 0)
        audio_bytes = None
        audio_format = None

        if has_audio:
            try:
                a_stream = container.streams.audio[0]
                codec_name = a_stream.codec_context.name
                audio_format = "adts" if codec_name == "aac" else codec_name

                out_buf = io.BytesIO()
                out_container = av.open(out_buf, mode="w", format=audio_format)
                out_stream = out_container.add_stream_from_template(a_stream)

                for packet in container.demux(a_stream):
                    if packet.dts is not None:
                        packet.stream = out_stream
                        out_container.mux(packet)

                out_container.close()
                audio_bytes = out_buf.getvalue() or None
                if not audio_bytes:
                    has_audio = False
            except Exception:
                has_audio = False
                audio_bytes = None
                audio_format = None

        container.close()

        def pyav_generator():
            v_container = av.open(video_path)
            vs = v_container.streams.video[0]
            vs.thread_type = "AUTO"
            raw_idx = 0
            yielded = 0
            try:
                for frame in v_container.decode(vs):
                    is_key = bool(frame.key_frame)
                    if sample_step == 1 or raw_idx == 0 or (raw_idx % sample_step == 0) or is_key:
                        rgb_arr = frame.to_ndarray(format="rgb24")
                        yield raw_idx, rgb_arr, is_key
                        yielded += 1
                    raw_idx += 1
                    if max_frames and yielded >= max_frames:
                        break
            finally:
                v_container.close()

        return {
            "fps": fps,
            "width": w,
            "height": h,
            "channels": c,
            "estimated_frames": est_frames,
            "sample_step": sample_step,
            "has_audio": has_audio,
            "audio_bytes": audio_bytes,
            "audio_format": audio_format,
            "generator": pyav_generator,
            "backend": "pyav"
        }

    @classmethod
    def _stream_with_cv2(
        cls,
        video_path: str,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else (target_fps or 24.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        sample_step = max(1, int(round(fps / target_fps))) if (target_fps and 0 < target_fps < fps) else 1

        def cv2_generator():
            vcap = cv2.VideoCapture(video_path)
            raw_idx = 0
            yielded = 0
            try:
                while vcap.isOpened():
                    if sample_step > 1 and raw_idx != 0 and (raw_idx % sample_step != 0):
                        if not vcap.grab():
                            break
                        raw_idx += 1
                        continue
                    ret, bgr = vcap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield raw_idx, rgb, (raw_idx == 0)
                    raw_idx += 1
                    yielded += 1
                    if max_frames and yielded >= max_frames:
                        break
            finally:
                vcap.release()

        return {
            "fps": fps,
            "width": w,
            "height": h,
            "channels": 3,
            "estimated_frames": total_est,
            "sample_step": sample_step,
            "has_audio": False,
            "audio_bytes": None,
            "audio_format": None,
            "generator": cv2_generator,
            "backend": "cv2"
        }


class FastMediaStreamWriter:
    """
    O(1) constant-memory streaming video writer.
    Encodes and writes frames incrementally via PyAV (with lossless audio remux)
    or OpenCV VideoWriter without buffering the full video in RAM.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        channels: int,
        fps: float,
        audio_bytes: Optional[bytes] = None,
        audio_format: Optional[str] = None
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.channels = channels
        self.fps = fps
        self.audio_bytes = audio_bytes
        self.audio_format = audio_format
        self._use_av = False
        self._container = None
        self._v_stream = None
        self._cv_writer = None

        self.is_container_lossless = False

        if HAS_AV:
            ext = os.path.splitext(output_path)[1].lower()
            candidates = []
            if ext in (".mkv", ".avi"):
                pix = "bgr0" if channels == 3 else ("bgra" if channels == 4 else "gray")
                candidates.append(("ffv1", pix, {}))
            if ext == ".mp4":
                if channels == 3:
                    candidates.append(("libx264rgb", "rgb24", {"crf": "0", "preset": "ultrafast"}))
                candidates.append(("h264", "yuv420p", {"crf": "17", "preset": "fast"}))

            for codec_name, pix_fmt, opts in candidates:
                try:
                    self._container = av.open(output_path, mode="w")
                    self._v_stream = self._container.add_stream(codec_name, rate=max(1, int(round(fps))))
                    self._v_stream.width = width
                    self._v_stream.height = height
                    self._v_stream.pix_fmt = pix_fmt
                    if opts:
                        self._v_stream.options = opts
                    self._use_av = True
                    self.is_container_lossless = codec_name in ("ffv1", "libx264rgb")
                    return
                except Exception:
                    if self._container is not None:
                        try:
                            self._container.close()
                        except Exception:
                            pass
                    self._container = None
                    self._v_stream = None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._cv_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write_frame(self, arr: np.ndarray):
        """Encodes and writes a single RGB/Grayscale uint8 frame."""
        if self._use_av and self._container is not None and self._v_stream is not None:
            fmt = "rgb24" if self.channels == 3 else ("rgba" if self.channels == 4 else "gray")
            av_frame = av.VideoFrame.from_ndarray(arr, format=fmt)
            for packet in self._v_stream.encode(av_frame):
                self._container.mux(packet)
        elif self._cv_writer is not None:
            if self.channels == 3:
                self._cv_writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            elif self.channels == 4:
                self._cv_writer.write(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR))
            else:
                self._cv_writer.write(arr)

    def close(self):
        """Flushes encoder, remuxes audio if present, and closes the container."""
        if self._use_av and self._container is not None and self._v_stream is not None:
            for packet in self._v_stream.encode():
                self._container.mux(packet)
            if self.audio_bytes and self.audio_format:
                try:
                    audio_reader = av.open(io.BytesIO(self.audio_bytes), format=self.audio_format)
                    if audio_reader.streams.audio:
                        in_a = audio_reader.streams.audio[0]
                        out_a = self._container.add_stream_from_template(in_a)
                        for packet in audio_reader.demux(in_a):
                            if packet.dts is not None:
                                packet.stream = out_a
                                self._container.mux(packet)
                    audio_reader.close()
                except Exception:
                    pass
            self._container.close()
            self._container = None
            self._v_stream = None
        elif self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None


class FastMediaWriter:
    """
    High-throughput media writer with lossless audio remuxing support via PyAV,
    with fallback to OpenCV VideoWriter.
    """

    @classmethod
    def write_video(
        cls,
        output_path: str,
        frames: List[np.ndarray],
        fps: float,
        audio_bytes: Optional[bytes] = None,
        audio_format: Optional[str] = None
    ):
        """
        Writes video frames to disk. If audio_bytes is provided and PyAV is available,
        remuxes the audio stream losslessly.
        """
        if not frames:
            return

        h, w = frames[0].shape[:2]
        c = frames[0].shape[2] if frames[0].ndim == 3 else 1
        stream_writer = FastMediaStreamWriter(
            output_path=output_path,
            width=w,
            height=h,
            channels=c,
            fps=fps,
            audio_bytes=audio_bytes,
            audio_format=audio_format
        )
        try:
            for arr in frames:
                stream_writer.write_frame(arr)
        finally:
            stream_writer.close()
