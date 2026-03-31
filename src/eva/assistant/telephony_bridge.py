"""Backward-compatible exports for the external agent bridge implementation."""

from eva.assistant.external.base import (
    BaseSegmentTranscriber,
    BaseTelephonyTransport,
    DeepgramSegmentTranscriber,
    NoopSegmentTranscriber,
    create_segment_transcriber,
)
from eva.assistant.external.bridge import (
    ExternalAgentBridgeServer,
    TelephonyBridgeServer,
    _SessionState,
    pcm16k_to_pcm24k as _pcm16k_to_pcm24k,
    pcm_to_wav_bytes as _pcm_to_wav_bytes,
)
from eva.models.config import TelephonyBridgeConfig

__all__ = [
    "BaseSegmentTranscriber",
    "BaseTelephonyTransport",
    "DeepgramSegmentTranscriber",
    "ExternalAgentBridgeServer",
    "NoopSegmentTranscriber",
    "TelephonyBridgeConfig",
    "TelephonyBridgeServer",
    "_SessionState",
    "_pcm16k_to_pcm24k",
    "_pcm_to_wav_bytes",
    "create_segment_transcriber",
]
