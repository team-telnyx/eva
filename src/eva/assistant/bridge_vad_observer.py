"""Real-time VAD observer for telephony bridge audio streams."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

from eva.assistant.agentic.audit_log import AuditLog
from eva.utils.logging import get_logger

logger = get_logger(__name__)

BRIDGE_SAMPLE_RATE = 16000
USER_EVENT_ROLE = "elevenlabs_user"
ASSISTANT_EVENT_ROLE = "pipecat_agent"
USER_SPEECH_SOURCE = "elevenlabs_agent"
ASSISTANT_SPEECH_SOURCE = "pipecat_assistant"


class SegmentTranscriber(Protocol):
    """Protocol for segment transcribers used by the bridge observer."""

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        """Return text for the provided audio segment."""


@dataclass(slots=True)
class _StreamState:
    """Per-stream VAD and buffering state."""

    role: str
    speech_event_type: str
    speech_source: str
    transcript_role: str
    audit_append: Callable[[str, str], None]
    vad: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_state: VADState = VADState.QUIET
    buffered_audio: bytearray | None = None
    speech_started_at: float | None = None


class BridgeVADObserver:
    """Observe bridge PCM streams and emit ElevenLabs-style logs in real time."""

    def __init__(
        self,
        output_dir: Path,
        transcriber: SegmentTranscriber,
        audit_log: AuditLog,
        *,
        vad_factory: Callable[[], Any] | None = None,
        time_provider: Callable[[], float] = time.time,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._transcriber = transcriber
        self._audit_log = audit_log
        self._time_provider = time_provider
        self._vad_factory = vad_factory or self._create_vad

        self._sequence = 0
        self._event_lock = asyncio.Lock()
        self._transcription_tasks: set[asyncio.Task[None]] = set()
        self._transcript_entries: list[dict[str, Any]] = []
        self._response_latencies: list[float] = []
        self._pending_user_speech_end: float | None = None
        self._finalized = False

        self._elevenlabs_path = self.output_dir / "elevenlabs_events.jsonl"
        self._elevenlabs_path.write_text("", encoding="utf-8")
        self._elevenlabs_file = open(self._elevenlabs_path, "a", encoding="utf-8")

        # External telephony assistants do not produce local pipecat TTS logs.
        self._pipecat_logs_path = self.output_dir / "pipecat_logs.jsonl"
        self._pipecat_logs_path.write_text("", encoding="utf-8")

        self._user_stream = _StreamState(
            role=USER_EVENT_ROLE,
            speech_event_type="user_speech",
            speech_source=USER_SPEECH_SOURCE,
            transcript_role="user",
            audit_append=lambda text, ts: self._audit_log.append_user_input(text, timestamp_ms=ts),
            vad=self._build_vad(),
        )
        self._assistant_stream = _StreamState(
            role=ASSISTANT_EVENT_ROLE,
            speech_event_type="assistant_speech",
            speech_source=ASSISTANT_SPEECH_SOURCE,
            transcript_role="assistant",
            audit_append=lambda text, ts: self._audit_log.append_assistant_output(text, timestamp_ms=ts),
            vad=self._build_vad(),
        )

        self._log_connection_state_sync("connected")
        self._log_connection_state_sync("session_started")

    @staticmethod
    def _create_vad() -> SileroVADAnalyzer:
        vad = SileroVADAnalyzer(
            sample_rate=BRIDGE_SAMPLE_RATE,
            params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.3,
                min_volume=0.6,
            ),
        )
        vad.set_sample_rate(BRIDGE_SAMPLE_RATE)
        return vad

    def _build_vad(self) -> Any:
        vad = self._vad_factory()
        if hasattr(vad, "set_sample_rate"):
            vad.set_sample_rate(BRIDGE_SAMPLE_RATE)
        return vad

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _write_event_now(self, event: dict[str, Any]) -> None:
        self._elevenlabs_file.write(json.dumps(event) + "\n")
        self._elevenlabs_file.flush()

    def _log_connection_state_sync(self, state: str, details: dict[str, Any] | None = None) -> None:
        ts = self._time_provider()
        self._write_event_now(
            {
                "timestamp": int(ts * 1000),
                "sequence": self._next_sequence(),
                "type": "connection_state",
                "data": {
                    "state": state,
                    "details": details or {},
                },
            }
        )

    async def _log_connection_state(self, state: str, details: dict[str, Any] | None = None) -> None:
        ts = self._time_provider()
        async with self._event_lock:
            self._write_event_now(
                {
                    "timestamp": int(ts * 1000),
                    "sequence": self._next_sequence(),
                    "type": "connection_state",
                    "data": {
                        "state": state,
                        "details": details or {},
                    },
                }
            )

    async def _log_audio_boundary(self, event_type: str, role: str, timestamp: float) -> None:
        async with self._event_lock:
            self._write_event_now(
                {
                    "timestamp": int(timestamp * 1000),
                    "sequence": self._next_sequence(),
                    "event_type": event_type,
                    "user": role,
                    "audio_timestamp": timestamp,
                }
            )

    async def _log_speech_event(self, event_type: str, text: str, source: str) -> None:
        ts = self._time_provider()
        async with self._event_lock:
            self._write_event_now(
                {
                    "timestamp": int(ts * 1000),
                    "sequence": self._next_sequence(),
                    "type": event_type,
                    "data": {
                        "text": text,
                        "source": source,
                    },
                }
            )

    def _schedule_transcription(self, stream: _StreamState, audio_data: bytes, speech_started_at: float) -> None:
        task = asyncio.create_task(self._transcribe_segment(stream, audio_data, speech_started_at))
        self._transcription_tasks.add(task)
        task.add_done_callback(self._transcription_tasks.discard)

    async def _transcribe_segment(self, stream: _StreamState, audio_data: bytes, speech_started_at: float) -> None:
        try:
            text = (await self._transcriber.transcribe(audio_data, BRIDGE_SAMPLE_RATE)).strip()
        except Exception as exc:
            logger.warning("Bridge segment transcription failed for %s: %s", stream.role, exc)
            return

        if not text:
            return

        await self._log_speech_event(stream.speech_event_type, text, stream.speech_source)

        timestamp_ms = str(int(speech_started_at * 1000))
        stream.audit_append(text, timestamp_ms)
        self._transcript_entries.append(
            {
                "timestamp_ms": int(speech_started_at * 1000),
                "timestamp": datetime.fromtimestamp(speech_started_at, UTC).isoformat(),
                "role": stream.transcript_role,
                "content": text,
            }
        )

    async def feed_user_audio(self, pcm_bytes: bytes, timestamp: float) -> None:
        """Process a user PCM chunk at an epoch-seconds timestamp."""
        await self._feed_stream(self._user_stream, pcm_bytes, timestamp)

    async def feed_assistant_audio(self, pcm_bytes: bytes, timestamp: float) -> None:
        """Process an assistant PCM chunk at an epoch-seconds timestamp."""
        await self._feed_stream(self._assistant_stream, pcm_bytes, timestamp)

    async def _feed_stream(self, stream: _StreamState, pcm_bytes: bytes, timestamp: float) -> None:
        if self._finalized or not pcm_bytes:
            return

        async with stream.lock:
            state = await stream.vad.analyze_audio(pcm_bytes)

            if state != VADState.QUIET and stream.buffered_audio is None:
                stream.buffered_audio = bytearray()

            if stream.buffered_audio is not None:
                stream.buffered_audio.extend(pcm_bytes)

            if state == VADState.SPEAKING and stream.last_state in {VADState.QUIET, VADState.STARTING}:
                stream.speech_started_at = timestamp
                await self._log_audio_boundary("audio_start", stream.role, timestamp)
                if stream.role == ASSISTANT_EVENT_ROLE and self._pending_user_speech_end is not None:
                    latency = round(timestamp - self._pending_user_speech_end, 4)
                    if latency > 0:
                        self._response_latencies.append(latency)
                    self._pending_user_speech_end = None

            if state == VADState.QUIET and stream.last_state in {VADState.SPEAKING, VADState.STOPPING}:
                segment_audio = bytes(stream.buffered_audio or b"")
                speech_started_at = stream.speech_started_at
                stream.buffered_audio = None
                stream.speech_started_at = None

                await self._log_audio_boundary("audio_end", stream.role, timestamp)
                if stream.role == USER_EVENT_ROLE:
                    self._pending_user_speech_end = timestamp

                if speech_started_at is not None and segment_audio:
                    self._schedule_transcription(stream, segment_audio, speech_started_at)
            elif state == VADState.QUIET and stream.last_state == VADState.STARTING:
                stream.buffered_audio = None
                stream.speech_started_at = None

            stream.last_state = state

    async def _finalize_stream(self, stream: _StreamState, timestamp: float) -> None:
        async with stream.lock:
            if stream.last_state in {VADState.SPEAKING, VADState.STOPPING} and stream.speech_started_at is not None:
                segment_audio = bytes(stream.buffered_audio or b"")
                speech_started_at = stream.speech_started_at

                await self._log_audio_boundary("audio_end", stream.role, timestamp)
                if stream.role == USER_EVENT_ROLE:
                    self._pending_user_speech_end = timestamp

                if segment_audio:
                    self._schedule_transcription(stream, segment_audio, speech_started_at)

            stream.buffered_audio = None
            stream.speech_started_at = None
            stream.last_state = VADState.QUIET

    async def finalize(
        self,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Flush pending segments and write final metrics artifacts."""
        if self._finalized:
            return
        self._finalized = True

        timestamp = self._time_provider()
        await self._finalize_stream(self._user_stream, timestamp)
        await self._finalize_stream(self._assistant_stream, timestamp)

        if self._transcription_tasks:
            await asyncio.gather(*list(self._transcription_tasks), return_exceptions=True)

        final_details = dict(details or {})
        if reason and "reason" not in final_details:
            final_details["reason"] = reason

        await self._log_connection_state("session_ended", final_details)
        self._write_response_latencies()

        self._elevenlabs_file.flush()
        self._elevenlabs_file.close()

    def _write_response_latencies(self) -> None:
        latencies_path = self.output_dir / "response_latencies.json"
        latencies_data = {
            "latencies": self._response_latencies,
            "mean": round(sum(self._response_latencies) / len(self._response_latencies), 4)
            if self._response_latencies
            else 0,
            "max": round(max(self._response_latencies), 4) if self._response_latencies else 0,
            "min": round(min(self._response_latencies), 4) if self._response_latencies else 0,
            "count": len(self._response_latencies),
            "source": "vad_observer",
        }
        with open(latencies_path, "w", encoding="utf-8") as file_obj:
            json.dump(latencies_data, file_obj, indent=2)

    def get_transcript_entries(self) -> list[dict[str, str]]:
        """Return transcribed user/assistant entries sorted chronologically."""
        sorted_entries = sorted(self._transcript_entries, key=lambda entry: entry["timestamp_ms"])
        return [
            {
                "timestamp": entry["timestamp"],
                "role": entry["role"],
                "content": entry["content"],
            }
            for entry in sorted_entries
        ]
