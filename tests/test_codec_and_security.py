"""
Automated Roundtrip, Integrity, 8-Bit Validation, Decoder Bounds & MCP Security Test Suite.
Runs without heavy model downloads using deterministic core codecs.
"""

import base64
import os
import logging
import numpy as np
import cv2
import pytest
import zstandard as zstd

from imgstd_project.core.encoder import IMGSTDEncoder, encode_image
from imgstd_project.core.decoder import IMGSTDDecoder, IntegrityError, decode_imgstd, MAX_WIDTH
from imgstd_project.core.inspector import inspect_imgstd
from imgstd_project.core.extractor import FastColorSegmenter, RLECodec
from imgstd_project.core.video_codec import encode_video_to_vidstd, decode_vidstd_to_video, extract_vidstd_frame, inspect_vidstd
from imgstd_project.core.fast_b85 import b85encode
from imgstd_project.server import mcp_server


class TestLosslessRoundtrip:
    def test_rgb_synthetic_and_random_roundtrip_sha_exact(self):
        rng = np.random.default_rng(42)
        for img in [
            np.tile(np.arange(64, dtype=np.uint8)[:, None, None], (1, 64, 3)),
            rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8),
        ]:
            enc = IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04")
            code, enc_m = enc.encode(img)
            assert code.startswith("IMGSTD::V04")
            insp = inspect_imgstd(code)
            assert insp["dimensions"]["width"] == img.shape[1]
            assert insp["dimensions"]["height"] == img.shape[0]
            assert insp["inspection_time_ms"] < 100.0

            recon, dec_m = decode_imgstd(code, verify_sha=True)
            assert dec_m["verified"] is True
            assert dec_m["sha256_match"] is True
            assert dec_m["mse"] == 0.0
            assert np.array_equal(img, recon)

    def test_grayscale_and_rgba_roundtrip(self):
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(64, 64, 1), dtype=np.uint8)
        rgba = rng.integers(0, 256, size=(64, 64, 4), dtype=np.uint8)

        for arr in (gray, rgba):
            enc = IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04")
            code, _ = enc.encode(arr)
            recon, m = decode_imgstd(code)
            assert m["mse"] == 0.0
            assert np.array_equal(arr, recon)

    def test_v01_legacy_roundtrip(self):
        from imgstd_project.core.legacy_v01 import LegacyV01Encoder
        rng = np.random.default_rng(9)
        img = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        code, _ = LegacyV01Encoder().encode(img)
        assert code.startswith("IMGSTD::V01")
        recon, m = decode_imgstd(code)
        assert m["verified"] is True
        assert np.array_equal(img, recon)

    def test_file_io_and_vidstd_roundtrip(self, tmp_path):
        img_path = tmp_path / "sample.png"
        out_stream = tmp_path / "sample.imgstd"
        out_png = tmp_path / "restored.png"

        img = np.zeros((64, 80, 3), dtype=np.uint8)
        img[:, :40] = [20, 120, 220]
        img[20:45, 30:70] = [200, 60, 30]
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        enc = IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04")
        code, _ = enc.encode(str(img_path))
        out_stream.write_text(code, encoding="utf-8")

        insp = inspect_imgstd(str(out_stream))
        assert insp["region_count"] >= 1
        recon, m = decode_imgstd(str(out_stream), output_image_path=str(out_png))
        assert m["verified"] is True
        assert np.array_equal(img, recon)
        assert out_png.exists()

        # Compact video stream roundtrip (I, S, P frames)
        f0 = np.full((64, 64, 3), 40, dtype=np.uint8)
        f1 = f0.copy()
        f2 = f0.copy()
        f2[16:48, 16:48] = [180, 90, 45]
        vid_path = tmp_path / "clip.vidstd"
        v_code, v_m = encode_video_to_vidstd([f0, f1, f2], output_path=str(vid_path), gop_size=2)
        assert v_m["total_frames"] == 3
        v_insp = inspect_vidstd(str(vid_path))
        assert v_insp["video_metadata"]["scene_count"] >= 1
        frame2, f2_m = extract_vidstd_frame(str(vid_path), 2)
        assert f2_m["verified"] is True
        assert np.array_equal(f2, frame2)
        frames, dec_vm = decode_vidstd_to_video(str(vid_path))
        assert dec_vm["all_frames_verified"] is True
        assert dec_vm["mse"] == 0.0
        assert np.array_equal(frames[0], f0) and np.array_equal(frames[2], f2)


class TestBitDepthAndErrorVisibility:
    def test_16bit_image_array_and_file_rejected(self, tmp_path):
        arr16 = np.full((32, 32, 3), 40000, dtype=np.uint16)
        enc = IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04")
        with pytest.raises(ValueError, match="8-bit"):
            enc.encode(arr16)

        png16 = tmp_path / "depth16.png"
        cv2.imwrite(str(png16), arr16)
        with pytest.raises(ValueError, match="8-bit"):
            enc.encode(str(png16))

    def test_vlm_failure_logs_warning_instead_of_silent_pass(self, caplog, monkeypatch):
        from imgstd_project.core import encoder as enc_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated caption crash")

        monkeypatch.setattr(enc_mod, "get_vlm_captioner", _boom)
        enc = IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04")
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        with caplog.at_level(logging.WARNING, logger="imgstd.encoder"):
            code, _ = enc.encode(img)
        assert code.startswith("IMGSTD::V04")
        assert any("VLM/CLIP captioning failed" in rec.message for rec in caplog.records)


class TestDecoderSecurityAndCorruption:
    def _make_valid_code(self):
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        return IMGSTDEncoder(extractor=FastColorSegmenter(), version="V04").encode(img)[0]

    def test_tampered_payload_triggers_integrity_error(self):
        code = self._make_valid_code()
        tampered = code.replace("SHA256:", "SHA256:0000")[:-10] if False else code.replace("SHA256:", "SHA256:ff")
        # Keep line length structure clean by corrupting hash hex
        idx = code.index("SHA256:") + 7
        bad_char = "0" if code[idx] != "0" else "1"
        tampered = code[:idx] + bad_char + code[idx + 1:]
        with pytest.raises(IntegrityError):
            decode_imgstd(tampered, verify_sha=True)

    def test_oversized_dimensions_rejected_before_decompress(self):
        code = self._make_valid_code()
        forged = code.replace("W:32\n", f"W:{MAX_WIDTH + 1}\n")
        with pytest.raises(ValueError, match="dimensions"):
            decode_imgstd(forged)

    def test_out_of_range_label_id_rejected(self):
        code = self._make_valid_code()
        corrupted = code.replace("REG:001|", "REG:999|", 1)
        with pytest.raises(ValueError, match="Out-of-range label_id"):
            decode_imgstd(corrupted, verify_sha=False)

    def test_zstd_decompression_bomb_blocked(self):
        code = self._make_valid_code()
        cctx = zstd.ZstdCompressor(level=3)
        bomb_b85 = b85encode(cctx.compress(b"\x00" * 100_000))
        res_s = code.index("[RES]\n") + 6
        res_e = code.index("\n[/RES]")
        forged = code[:res_s] + bomb_b85 + code[res_e:]
        with pytest.raises((zstd.ZstdError, ValueError)):
            decode_imgstd(forged, verify_sha=False)


class TestMCPSandboxAndVLMTrust:
    def test_mcp_blocks_path_traversal_and_outside_reads(self, tmp_path, monkeypatch):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("TOP_SECRET", encoding="utf-8")

        monkeypatch.setenv("IMGSTD_SANDBOX_ROOT", str(sandbox))

        with pytest.raises(PermissionError):
            mcp_server.resolve_sandbox_path("../secret.txt")
        with pytest.raises(PermissionError):
            mcp_server.resolve_sandbox_path(str(outside))
        with pytest.raises(PermissionError):
            mcp_server.inspect_imgstd_code("../secret.txt")
        with pytest.raises(PermissionError):
            mcp_server.encode_image_to_imgstd(str(outside))

        # Inside sandbox works normally
        img_in = sandbox / "ok.png"
        cv2.imwrite(str(img_in), np.full((32, 32, 3), 90, dtype=np.uint8))
        out_res = mcp_server.encode_image_to_imgstd("ok.png", "ok.imgstd")
        assert os.path.realpath(out_res["output_path"]).startswith(os.path.realpath(str(sandbox)))
        recolor = mcp_server.recolor_imgstd_region("ok.imgstd", "001", [10, 20, 30])
        assert recolor["success"] is True

    def test_no_trust_remote_code_in_vlm_captioner(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "imgstd_project", "core", "vlm_captioner.py",
        )
        src = open(path, "r", encoding="utf-8").read()
        assert "trust_remote_code=True" not in src
        assert "os.getcwd()" not in src


class TestStartupCacheFastPaths:
    def test_yolo_world_text_feature_cache_skips_repeat_clip(self, tmp_path, monkeypatch):
        import types
        import torch
        from imgstd_project.core import config as cfg_mod
        from imgstd_project.core.extractor import _set_world_classes_cached

        monkeypatch.setattr(cfg_mod, "get_weights_root", lambda: str(tmp_path))

        calls = {"count": 0}

        class DummyWorld:
            def __init__(self):
                self.device = "cpu"
                self.predictor = "stale"
                self.model = types.SimpleNamespace(
                    txt_feats=None,
                    names=[],
                    model=[types.SimpleNamespace(nc=0)],
                )

            def set_classes(self, classes):
                calls["count"] += 1
                self.model.txt_feats = torch.ones((1, len(classes), 4), dtype=torch.float32)
                self.model.model[-1].nc = len(classes)
                self.model.names = list(classes)
                self.predictor = None

        classes = ["soldier", "rodent"]
        w1 = DummyWorld()
        _set_world_classes_cached(w1, classes)
        assert calls["count"] == 1

        w2 = DummyWorld()
        _set_world_classes_cached(w2, classes)
        assert calls["count"] == 1  # Hit cache, no second CLIP encoding
        assert w2.model.names == classes
        assert w2.model.model[-1].nc == 2
        assert w2.model.txt_feats.shape == (1, 2, 4)
        assert w2.predictor is None

    def test_mobileclip_visual_meta_and_torchscript_cache(self, tmp_path):
        import json
        from imgstd_project.core.scene_clip import MobileCLIPSceneAnalyzer

        analyzer = MobileCLIPSceneAnalyzer("mobileclip2-s0", weights_dir=str(tmp_path))
        meta_path = tmp_path / "visual_meta.json"
        analyzer._image_size = 256
        analyzer._image_mean = (0.1, 0.2, 0.3)
        analyzer._image_std = (0.4, 0.5, 0.6)
        analyzer._save_visual_meta(str(meta_path))

        loaded = MobileCLIPSceneAnalyzer("mobileclip2-s0", weights_dir=str(tmp_path))
        loaded._image_size = 128
        assert loaded._load_visual_meta(str(meta_path)) is True
        assert loaded._image_size == 256
        assert loaded._image_mean == (0.1, 0.2, 0.3)
        assert loaded._image_std == (0.4, 0.5, 0.6)

        # Wrong model signature must reject stale metadata
        wrong = MobileCLIPSceneAnalyzer("mobileclip2-s2", weights_dir=str(tmp_path))
        assert wrong._load_visual_meta(str(meta_path)) is False

    def test_zero_torch_import_when_onnx_caches_exist(self, tmp_path):
        """When ONNX caches are populated in weights_dir, standard V04 extractors load ORT without PyTorch."""
        import hashlib, subprocess, sys
        from imgstd_project.core.config import get_weights_root
        from imgstd_project.core.extractor import YoloWorldSemanticExtractor

        sig = hashlib.sha256("\n".join(YoloWorldSemanticExtractor.HYBRID_OPEN_CLASSES).encode("utf-8")).hexdigest()[:16]
        mc_onnx = os.path.join(get_weights_root(), "MobileCLIP2-S0", "visual_model.onnx")
        world_onnx = os.path.join(get_weights_root(), f"yolov8s-worldv2_hybrid_{sig}.onnx")
        if not (os.path.exists(mc_onnx) and os.path.exists(world_onnx)):
            pytest.skip("Local ONNX caches not yet generated on this runner.")

        out_file = str(tmp_path / "ort_test.imgstd")
        img_file = str(tmp_path / "ort_in.png")
        cv2.imwrite(img_file, np.full((256, 256, 3), 120, dtype=np.uint8))

        code = (
            "import sys; "
            "from imgstd_project.core.encoder import encode_image; "
            f"encode_image(r'{img_file}', output_path=r'{out_file}', enable_smolvlm2=False); "
            "assert 'torch' not in sys.modules, 'Unexpected PyTorch import in ONNX path!'; "
            "assert 'open_clip' not in sys.modules"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

    def test_event_calibration_and_vlm_resize_helper(self):
        from imgstd_project.core.vlm_captioner import SmolVLM2Captioner
        from imgstd_project.core.embedding import RegionEmbeddingGenerator

        # Calm baseline must suppress weak noisy event hits
        assert SmolVLM2Captioner.format_event_tag({"event": "calm", "event_sim": 0.30, "event_calm_sim": 0.30}) is None
        assert SmolVLM2Captioner.format_event_tag({
            "event": "combat or physical confrontation", "event_sim": 0.22, "event_calm_sim": 0.21
        }) is None
        # Clear event margin above calm passes
        assert SmolVLM2Captioner.format_event_tag({
            "event": "fire and rising smoke", "event_sim": 0.27, "event_calm_sim": 0.19
        }) == "fire and rising smoke"

        # Bounded resize helper prevents 1080p multi-tile explosion in SmolVLM2
        big_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pil_resized = SmolVLM2Captioner._prepare_vlm_pil(big_img, max_side=512)
        assert max(pil_resized.size) == 512

        # Strided embedding preserves shape & dtype cleanly on 1080p crops
        emb = RegionEmbeddingGenerator.compute(big_img, (0, 0, 1920, 1080))
        assert emb.shape == (32,) and emb.dtype == np.int8
