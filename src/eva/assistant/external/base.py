"""Shared abstractions for hosted external voice agent integrations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from eva.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from eva.models.config import ExternalAgentConfig

logger = get_logger(__name__)


class BaseSegmentTranscriber(ABC):
    """Transcribe PCM audio segments into text."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        """Return text for the provided audio segment."""


class NoopSegmentTranscriber(BaseSegmentTranscriber):
    """Fallback transcriber when no STT service is configured."""

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        return ""


class DeepgramSegmentTranscriber(BaseSegmentTranscriber):
    """Deepgram prerecorded transcription client."""

    def __init__(self, params: dict[str, Any]):
        self._params = dict(params)
        self._api_key = self._params["api_key"]
        self._endpoint = self._params.get("url", "https://api.deepgram.com/v1/listen")

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        if not audio_data:
            return ""

        request_params = {
            "model": self._params.get("model", "nova-2"),
            "smart_format": self._params.get("smart_format", "true"),
            "punctuate": self._params.get("punctuate", "true"),
        }
        if language := self._params.get("language"):
            request_params["language"] = language

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }

        from eva.assistant.external.bridge import pcm_to_wav_bytes

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._endpoint,
                params=request_params,
                headers=headers,
                content=pcm_to_wav_bytes(audio_data, sample_rate=sample_rate),
            )
            response.raise_for_status()

        payload = response.json()
        return (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
            .strip()
        )


def create_segment_transcriber(config: "ExternalAgentConfig") -> BaseSegmentTranscriber:
    """Create a segment transcriber for transcript.jsonl output."""
    if not config.stt:
        return NoopSegmentTranscriber()
    if config.stt.lower().startswith("deepgram"):
        return DeepgramSegmentTranscriber(config.stt_params)

    logger.warning(
        "Unsupported external agent STT provider '%s', transcript.jsonl will be best-effort",
        config.stt,
    )
    return NoopSegmentTranscriber()


class BaseTelephonyTransport(ABC):
    """Abstract audio transport for external assistant connectivity."""

    def __init__(self, sip_uri: str, conversation_id: str, webhook_base_url: str):
        self.sip_uri = sip_uri
        self.conversation_id = conversation_id
        self.webhook_base_url = webhook_base_url
        self._audio_handler: Callable[[bytes], Awaitable[None]] | None = None

    @property
    def external_call_id(self) -> str | None:
        """Return the platform-specific call identifier, if available."""
        return None

    @property
    def call_leg_id(self) -> str | None:
        """Return the call leg identifier, if available."""
        return None

    @property
    def eva_call_id(self) -> str | None:
        """Return the EVA-generated routing identifier, if available."""
        return None

    def set_audio_handler(self, handler: Callable[[bytes], Awaitable[None]]) -> None:
        self._audio_handler = handler

    async def emit_audio(self, audio_data: bytes) -> None:
        if self._audio_handler is not None:
            await self._audio_handler(audio_data)

    @abstractmethod
    async def start(self) -> None:
        """Open the transport session to the external assistant."""

    @abstractmethod
    async def stop(self) -> None:
        """Close the transport session."""

    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Send 16kHz 16-bit PCM (L16) audio to the external assistant."""


class ExternalAgentProvider(ABC):
    """Plugin interface for hosted voice agent APIs."""

    @abstractmethod
    def create_transport(self, conversation_id: str, webhook_base_url: str) -> BaseTelephonyTransport:
        """Create a transport connection to the external agent."""

    @abstractmethod
    async def setup(self) -> None:
        """One-time setup before a benchmark run."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up after a benchmark run."""

    @abstractmethod
    async def fetch_intended_speech(self, transport: BaseTelephonyTransport) -> list[dict[str, Any]]:
        """Fetch intended assistant speech for post-call enrichment."""

    def register_webhook_routes(self, app: "FastAPI") -> None:
        """Register provider-specific webhook routes."""

    def on_transport_started(self, transport: BaseTelephonyTransport) -> None:
        """Called after transport.start() completes."""

    async def get_active_transport(self, conversation_id: str) -> BaseTelephonyTransport | None:
        """Return the active provider transport for a media stream connection."""
        return None

    def get_tool_call_route_id(self, transport: BaseTelephonyTransport) -> str | None:
        """Return the route identifier used in `/tools/{call_id}/...` webhook paths."""
        return transport.eva_call_id

    def register_record_id(self, route_id: str, record_id: str) -> None:
        """Associate provider-specific metadata with a route ID."""

