"""Bridge server for benchmarking hosted external voice assistants."""

import asyncio
import base64
import io
import json
import time
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.external.base import (
    BaseSegmentTranscriber,
    BaseTelephonyTransport,
    ExternalAgentProvider,
    create_segment_transcriber,
)
from eva.assistant.external.bridge_vad_observer import BridgeVADObserver
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.models.agents import AgentConfig
from eva.models.config import ExternalAgentConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2


def pcm16k_to_pcm24k(pcm_16khz: bytes) -> bytes:
    """Upsample 16kHz 16-bit PCM audio to 24kHz 16-bit PCM."""
    if not pcm_16khz:
        return b""

    pcm_24khz, _ = audioop.ratecv(
        pcm_16khz,
        PCM_SAMPLE_WIDTH,
        1,
        INPUT_SAMPLE_RATE,
        OUTPUT_SAMPLE_RATE,
        None,
    )
    return pcm_24khz


def pcm_to_wav_bytes(audio_data: bytes, sample_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """Encode PCM audio as a mono WAV file."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return buffer.getvalue()


def _append_timed_audio(buffer: bytearray, audio_data: bytes, offset_seconds: float) -> None:
    """Mix audio into a PCM buffer using a monotonic time offset."""
    if not audio_data:
        return

    start_frame = max(0, round(offset_seconds * OUTPUT_SAMPLE_RATE))
    start_byte = start_frame * PCM_SAMPLE_WIDTH
    end_byte = start_byte + len(audio_data)

    if len(buffer) < end_byte:
        buffer.extend(b"\x00" * (end_byte - len(buffer)))

    existing = bytes(buffer[start_byte:end_byte])
    if existing and any(existing):
        buffer[start_byte:end_byte] = audioop.add(existing, audio_data, PCM_SAMPLE_WIDTH)
    else:
        buffer[start_byte:end_byte] = audio_data


@dataclass(slots=True)
class _SessionState:
    """Accumulated audio and timing state for a bridge session."""

    started_at: datetime
    user_audio: bytearray = field(default_factory=bytearray)
    assistant_audio: bytearray = field(default_factory=bytearray)
    mixed_audio: bytearray = field(default_factory=bytearray)

    def add_chunk(self, role: str, pcm_data: bytes, offset_seconds: float) -> None:
        pcm_24khz = pcm16k_to_pcm24k(pcm_data)
        if not pcm_24khz:
            return

        if role == "user":
            _append_timed_audio(self.user_audio, pcm_24khz, offset_seconds)
        else:
            _append_timed_audio(self.assistant_audio, pcm_24khz, offset_seconds)

        _append_timed_audio(self.mixed_audio, pcm_24khz, offset_seconds)


class ExternalAgentBridgeServer:
    """Bridge the user simulator websocket to an external hosted voice agent."""

    def __init__(
        self,
        current_date_time: str,
        bridge_config: ExternalAgentConfig,
        agent: AgentConfig,
        agent_config_path: str,
        scenario_db_path: str,
        output_dir: Path,
        port: int,
        conversation_id: str,
        provider: ExternalAgentProvider,
        segment_transcriber: BaseSegmentTranscriber | None = None,
        observer_factory: Callable[[Path, BaseSegmentTranscriber, AuditLog], BridgeVADObserver] | None = None,
    ):
        self.current_date_time = current_date_time
        self.bridge_config = bridge_config
        self.agent = agent
        self.agent_config_path = agent_config_path
        self.scenario_db_path = scenario_db_path
        self.output_dir = Path(output_dir)
        self.port = port
        self.conversation_id = conversation_id
        self.provider = provider

        self.audit_log = AuditLog()
        self.tool_handler = ToolExecutor(
            tool_config_path=agent_config_path,
            scenario_db_path=scenario_db_path,
            tool_module_path=self.agent.tool_module_path,
            current_date_time=self.current_date_time,
        )

        self._segment_transcriber = segment_transcriber or create_segment_transcriber(bridge_config)
        self._observer_factory = observer_factory or (
            lambda output_dir, transcriber, audit_log: BridgeVADObserver(output_dir, transcriber, audit_log)
        )
        self._session_state: _SessionState | None = None
        self._vad_observer: BridgeVADObserver | None = None
        self._session_started_monotonic: float | None = None
        self._session_end_reason: str | None = None
        self._transport: BaseTelephonyTransport | None = None
        self._completed_transport: BaseTelephonyTransport | None = None
        self._transport_started_callback: Callable[[BaseTelephonyTransport], Awaitable[None]] | None = None

        self._app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._running = False

        self._register_routes()

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI app for tests."""
        return self._app

    def set_transport_started_callback(
        self,
        callback: Callable[[BaseTelephonyTransport], Awaitable[None]],
    ) -> None:
        """Register a callback invoked after the provider transport starts."""
        self._transport_started_callback = callback

    async def start(self) -> None:
        """Start the bridge server."""
        if self._running:
            logger.warning("External agent bridge server already running")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._running = True

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info("External agent bridge started on ws://localhost:%s", self.port)

    async def stop(self) -> None:
        """Stop the bridge server and save outputs."""
        if not self._running:
            await self._save_outputs()
            return

        self._running = False

        if self._transport is not None:
            await self._transport.stop()
            self._transport = None

        if self._server:
            self._server.should_exit = True

        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass

        self._server = None
        self._server_task = None

        await self._save_outputs()
        logger.info("External agent bridge stopped on port %s", self.port)

    def _register_routes(self) -> None:
        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        @self._app.websocket("/")
        async def websocket_root(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

    async def _handle_session(self, websocket: WebSocket) -> None:
        """Bridge a user simulator websocket to the configured provider transport."""
        logger.info("External agent bridge client connected for %s", self.conversation_id)

        self._session_state = _SessionState(started_at=datetime.now(UTC))
        self._session_started_monotonic = asyncio.get_event_loop().time()
        self._session_end_reason = None
        self._transport = self.provider.create_transport(self.conversation_id, self.bridge_config.webhook_base_url)
        self._completed_transport = self._transport
        self._vad_observer = self._observer_factory(self.output_dir, self._segment_transcriber, self.audit_log)

        assistant_audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def on_transport_audio(audio_data: bytes) -> None:
            assert self._session_state is not None
            offset_seconds = self._elapsed_seconds()
            self._session_state.add_chunk("assistant", audio_data, offset_seconds)
            if self._vad_observer is not None:
                await self._vad_observer.feed_assistant_audio(audio_data, self._current_epoch_seconds())
            await assistant_audio_queue.put(audio_data)

        self._transport.set_audio_handler(on_transport_audio)

        async def send_assistant_audio() -> None:
            while True:
                audio_data = await assistant_audio_queue.get()
                if audio_data is None:
                    break
                await websocket.send_json(
                    {
                        "event": "media",
                        "conversation_id": self.conversation_id,
                        "media": {"payload": base64.b64encode(audio_data).decode("utf-8")},
                    }
                )

        sender_task = asyncio.create_task(send_assistant_audio())

        try:
            await self._transport.start()
            self.provider.on_transport_started(self._transport)
            if self._transport_started_callback is not None:
                await self._transport_started_callback(self._transport)

            disconnect_event = getattr(self._transport, "_disconnected_event", None)
            transport_done = asyncio.create_task(disconnect_event.wait()) if disconnect_event is not None else None

            try:
                while True:
                    ws_receive = asyncio.create_task(websocket.receive_text())
                    pending = [ws_receive]
                    if transport_done is not None:
                        pending.append(transport_done)
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                    if transport_done is not None and transport_done in done:
                        ws_receive.cancel()
                        self._session_end_reason = "assistant_hangup"
                        logger.info(
                            "External call ended for %s - ending session",
                            self.conversation_id,
                        )
                        try:
                            await websocket.send_json(
                                {
                                    "event": "stop",
                                    "conversation_id": self.conversation_id,
                                    "reason": "assistant_hangup",
                                }
                            )
                        except Exception:
                            pass
                        break

                    message = ws_receive.result()
                    payload = json.loads(message)
                    event = payload.get("event")

                    if event == "media":
                        audio_base64 = payload.get("media", {}).get("payload", "")
                        pcm_audio = base64.b64decode(audio_base64) if audio_base64 else b""
                        if pcm_audio:
                            assert self._session_state is not None
                            offset_seconds = self._elapsed_seconds()
                            self._session_state.add_chunk("user", pcm_audio, offset_seconds)
                            if self._vad_observer is not None:
                                await self._vad_observer.feed_user_audio(pcm_audio, self._current_epoch_seconds())
                            await self._transport.send_audio(pcm_audio)
                    elif event == "stop":
                        self._session_end_reason = payload.get("reason") or self._session_end_reason
                        logger.info("Received stop event for %s", self.conversation_id)
                        break
                    else:
                        logger.debug("Ignoring external bridge event '%s'", event)
            finally:
                if transport_done is not None:
                    transport_done.cancel()

        except WebSocketDisconnect:
            logger.info("External agent bridge websocket disconnected for %s", self.conversation_id)
        except Exception as exc:
            logger.error("External agent bridge session error: %s", exc, exc_info=True)
        finally:
            if self._session_state is not None:
                user_bytes = len(self._session_state.user_audio)
                asst_bytes = len(self._session_state.assistant_audio)
                logger.info(
                    "Session audio captured: user=%d bytes, assistant=%d bytes",
                    user_bytes,
                    asst_bytes,
                )

            await assistant_audio_queue.put(None)
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

            if self._transport is not None:
                await self._transport.stop()

            if self._vad_observer is not None:
                await self._vad_observer.finalize(reason=self._infer_session_end_reason())

            logger.info("External agent bridge client finished for %s", self.conversation_id)

    def _elapsed_seconds(self) -> float:
        if self._session_started_monotonic is None:
            return 0.0
        return max(0.0, asyncio.get_event_loop().time() - self._session_started_monotonic)

    @staticmethod
    def _current_epoch_seconds() -> float:
        return time.time()

    def _infer_session_end_reason(self) -> str | None:
        for entry in reversed(self.audit_log.transcript):
            if entry.get("message_type") not in {"tool_call", "tool_response"}:
                continue
            value = entry.get("value", {})
            if isinstance(value, dict) and value.get("tool") == "end_call":
                return "goodbye"

        return self._session_end_reason

    def _read_assistant_audio_start_timestamps(self) -> list[int]:
        """Read assistant audio_start timestamps for turn-aligned tts_text logs."""
        events_path = self.output_dir / "elevenlabs_events.jsonl"
        if not events_path.exists():
            return []
        timestamps: list[int] = []
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event_type") == "audio_start" and event.get("user") == "pipecat_agent":
                    ts = event.get("timestamp", 0)
                    if ts:
                        timestamps.append(ts)
        return timestamps

    async def _write_pipecat_tts_events(self, pipecat_logs_path: Path) -> None:
        intended_speech: list[dict[str, Any]] = []
        if self._completed_transport is not None:
            intended_speech = await self.provider.fetch_intended_speech(self._completed_transport)

        audio_start_timestamps = self._read_assistant_audio_start_timestamps()
        for i, item in enumerate(intended_speech):
            if i < len(audio_start_timestamps):
                item["timestamp_ms"] = audio_start_timestamps[i]
            elif audio_start_timestamps:
                last_ts = audio_start_timestamps[-1]
                offset = i - len(audio_start_timestamps) + 1
                item["timestamp_ms"] = last_ts + offset

        # Keep pipecat_logs.jsonl for upstream compatibility. Rename to
        # framework_logs.jsonl once the metrics pipeline no longer depends on it.
        with open(pipecat_logs_path, "w", encoding="utf-8") as file_obj:
            for i, item in enumerate(intended_speech):
                if i > 0:
                    file_obj.write(
                        json.dumps(
                            {
                                "type": "turn_end",
                                "timestamp": item["timestamp_ms"] - 1,
                                "start_timestamp": item["timestamp_ms"] - 1,
                                "data": {"frame": ""},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                file_obj.write(
                    json.dumps(
                        {
                            "type": "tts_text",
                            "timestamp": item["timestamp_ms"],
                            "start_timestamp": item["timestamp_ms"],
                            "data": {"frame": item["text"]},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    async def _save_outputs(self) -> None:
        """Persist bridge outputs in the same shape as AssistantServer."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving outputs to %s (session_state=%s, user_audio=%d, assistant_audio=%d)",
            self.output_dir,
            self._session_state is not None,
            len(self._session_state.user_audio) if self._session_state else 0,
            len(self._session_state.assistant_audio) if self._session_state else 0,
        )

        transcript_entries = self._vad_observer.get_transcript_entries() if self._vad_observer is not None else []
        transcript_path = self.output_dir / "transcript.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as transcript_file:
            for entry in transcript_entries:
                transcript_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        audit_path = self.output_dir / "audit_log.json"
        self.audit_log.save(audit_path)

        self._save_audio()

        initial_db_path = self.output_dir / "initial_scenario_db.json"
        with open(initial_db_path, "w", encoding="utf-8") as initial_db_file:
            json.dump(self.get_initial_scenario_db(), initial_db_file, indent=2, sort_keys=True, default=str)

        final_db_path = self.output_dir / "final_scenario_db.json"
        with open(final_db_path, "w", encoding="utf-8") as final_db_file:
            json.dump(self.get_final_scenario_db(), final_db_file, indent=2, sort_keys=True, default=str)

        pipecat_logs_path = self.output_dir / "pipecat_logs.jsonl"
        await self._write_pipecat_tts_events(pipecat_logs_path)

        response_latencies_path = self.output_dir / "response_latencies.json"
        if not response_latencies_path.exists():
            with open(response_latencies_path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "latencies": [],
                        "mean": 0,
                        "max": 0,
                        "min": 0,
                        "count": 0,
                        "source": "vad_observer",
                    },
                    file_obj,
                    indent=2,
                )

        logger.info("External agent bridge outputs saved to %s", self.output_dir)

    def _save_audio(self) -> None:
        if self._session_state is None:
            return

        self._save_wav_file(bytes(self._session_state.mixed_audio), self.output_dir / "audio_mixed.wav")
        self._save_wav_file(bytes(self._session_state.user_audio), self.output_dir / "audio_user.wav")
        self._save_wav_file(bytes(self._session_state.assistant_audio), self.output_dir / "audio_assistant.wav")

    @staticmethod
    def _save_wav_file(audio_data: bytes, file_path: Path) -> None:
        try:
            with wave.open(str(file_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
                wav_file.setframerate(OUTPUT_SAMPLE_RATE)
                wav_file.writeframes(audio_data)
        except Exception as exc:
            logger.error("Error saving audio to %s: %s", file_path, exc)

    def get_conversation_stats(self) -> dict[str, Any]:
        """Get pre-metric conversation statistics."""
        return self.audit_log.get_stats()

    def get_initial_scenario_db(self) -> dict[str, Any]:
        """Get the scenario DB state before tool execution."""
        return self.tool_handler.original_db

    def get_final_scenario_db(self) -> dict[str, Any]:
        """Get the scenario DB state after tool execution."""
        return self.tool_handler.db


TelephonyBridgeServer = ExternalAgentBridgeServer

__all__ = [
    "ExternalAgentBridgeServer",
    "TelephonyBridgeServer",
    "_SessionState",
    "pcm16k_to_pcm24k",
    "pcm_to_wav_bytes",
]
