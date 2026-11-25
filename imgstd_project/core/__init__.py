"""
IMGSTD Core Package: Standard Semantic-Lossless Image Codec Protocol (V02)
Supports both IMGSTD::V01 (legacy) and IMGSTD::V02 (current).
"""

from .extractor import RegionDescriptor, RLECodec, BaseSemanticExtractor, FastColorSegmenter
from .encoder import IMGSTDEncoder, encode_image
from .decoder import IMGSTDDecoder, decode_imgstd, IntegrityError
from .inspector import IMGSTDInspector, inspect_imgstd
from .verifier import IntegrityVerifier
from .legacy_v01 import LegacyV01Encoder, LegacyV01Decoder
from .video_codec import (
    VideoEncoder,
    VideoDecoder,
    VideoInspector,
    encode_video_to_vidstd,
    decode_vidstd_to_video,
    extract_vidstd_frame,
    inspect_vidstd,
)
from .video_tracker import SceneCutDetector, TemporalObjectTracker

__all__ = [
    # Extractor / Label Map
    "RegionDescriptor",
    "RLECodec",
    "BaseSemanticExtractor",
    "FastColorSegmenter",
    # V02 Encoder / Decoder
    "IMGSTDEncoder",
    "encode_image",
    "IMGSTDDecoder",
    "decode_imgstd",
    "IntegrityError",
    # Inspector (V01 + V02)
    "IMGSTDInspector",
    "inspect_imgstd",
    # Verifier
    "IntegrityVerifier",
    # Legacy V01 reference
    "LegacyV01Encoder",
    "LegacyV01Decoder",
    # VIDSTD Video Protocol
    "VideoEncoder",
    "VideoDecoder",
    "VideoInspector",
    "encode_video_to_vidstd",
    "decode_vidstd_to_video",
    "extract_vidstd_frame",
    "inspect_vidstd",
    "SceneCutDetector",
    "TemporalObjectTracker",
]
