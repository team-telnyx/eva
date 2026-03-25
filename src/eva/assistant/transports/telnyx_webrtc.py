"""Telnyx WebRTC transport backed by a Node.js helper process."""

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Mapping

import aiohttp

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.assistant.telephony_bridge import BaseTelephonyTransport, PCM_SAMPLE_WIDTH
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_HELPER_INSTALL_LOCK = asyncio.Lock()
_HELPER_READY_TIMEOUT_SECONDS = 30.0
_HELPER_CONNECT_RETRY_SECONDS = 0.1


def _transports_dir() -> Path:
    return Path(__file__).resolve().parent


def _helper_script_path() -> Path:
    return _transports_dir() / "telnyx_webrtc_helper.js"


def _node_modules_path() -> Path:
    return _transports_dir() / "node_modules"


async def _read_process_stream(stream: asyncio.StreamReader | None, level: str) -> None:
    if stream is None:
        return

    while line := await stream.readline():
        message = line.decode("utf-8", errors="replace").rstrip()
        if not message:
            continue

        if level == "error":
            logger.error(f"Telnyx helper: {message}")
        else:
            logger.info(f"Telnyx helper: {message}")


async def ensure_telnyx_webrtc_helper_dependencies() -> None:
    """Install Node.js dependencies for the Telnyx WebRTC helper when needed."""
    if _node_modules_path().exists():
        return

    async with _HELPER_INSTALL_LOCK:
        if _node_modules_path().exists():
            return

        logger.info(f"Installing Telnyx WebRTC helper dependencies in {_transports_dir()}")
        process = await asyncio.create_subprocess_exec(
            "npm",
            "install",
            "--no-audit",
            "--no-fund",
            cwd=str(_transports_dir()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_task = asyncio.create_task(_read_process_stream(process.stdout, "info"))
        stderr_task = asyncio.create_task(_read_process_stream(process.stderr, "error"))
        return_code = await process.wait()
        await asyncio.gather(stdout_task, stderr_task)

        if return_code != 0:
            raise RuntimeError(f"npm install failed for Telnyx WebRTC helper with exit code {return_code}")


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(probe.getsockname()[1])


class TelnyxWebRTCTransport(BaseTelephonyTransport):
    """WebRTC transport using Telnyx's SDK via a Node.js subprocess."""

    def __init__(
        self,
        assistant_id: str,
        conversation_id: str,
        webhook_base_url: str,
        ws_port: int | None = None,
        *,
        node_executable: str = "node",
        helper_script_path: Path | None = None,
        helper_extra_args: list[str] | None = None,
        helper_env: Mapping[str, str] | None = None,
        ready_timeout_seconds: float = _HELPER_READY_TIMEOUT_SECONDS,
        install_dependencies: bool = True,
    ):
        super().__init__(
            sip_uri=f"telnyx-assistant:{assistant_id}",
            conversation_id=conversation_id,
            webhook_base_url=webhook_base_url,
        )
        self.assistant_id = assistant_id
        self.ws_port = ws_port or _find_available_port()
        self.node_executable = node_executable
        self.helper_script_path = helper_script_path or _helper_script_path()
        self.helper_extra_args = helper_extra_args or []
        self.helper_env = dict(helper_env or {})
        self.ready_timeout_seconds = ready_timeout_seconds
        self.install_dependencies = install_dependencies

        self._helper_process: asyncio.subprocess.Process | None = None
        self._helper_stdout_task: asyncio.Task[None] | None = None
        self._helper_stderr_task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._ready_event = asyncio.Event()
        self._helper_error: RuntimeError | None = None

    async def start(self) -> None:
        if self._websocket is not None:
            logger.warning(f"Telnyx WebRTC transport already started for {self.assistant_id}")
            return

        if self.install_dependencies:
            await ensure_telnyx_webrtc_helper_dependencies()

        logger.info(f"Starting Telnyx WebRTC transport for assistant {self.assistant_id}")

        helper_command = [
            self.node_executable,
            str(self.helper_script_path),
            "--assistant-id",
            self.assistant_id,
            "--ws-port",
            str(self.ws_port),
            "--conversation-id",
            self.conversation_id,
            *self.helper_extra_args,
        ]
        helper_env = os.environ.copy()
        helper_env.update(self.helper_env)
        node_path = helper_env.get("NODE_PATH")
        helper_node_modules = str(_node_modules_path())
        helper_env["NODE_PATH"] = (
            f"{helper_node_modules}{os.pathsep}{node_path}" if node_path else helper_node_modules
        )

        self._helper_process = await asyncio.create_subprocess_exec(
            *helper_command,
            cwd=str(_transports_dir()),
            env=helper_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._helper_stdout_task = asyncio.create_task(_read_process_stream(self._helper_process.stdout, "info"))
        self._helper_stderr_task = asyncio.create_task(_read_process_stream(self._helper_process.stderr, "error"))

        self._session = aiohttp.ClientSession()
        self._websocket = await self._connect_to_helper()
        self._reader_task = asyncio.create_task(self._read_helper_messages())

        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=self.ready_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Timed out waiting for Telnyx WebRTC helper readiness for assistant {self.assistant_id}",
            ) from exc

        if self._helper_error is not None:
            raise self._helper_error

    async def _connect_to_helper(self) -> aiohttp.ClientWebSocketResponse:
        assert self._session is not None
        helper_url = f"ws://127.0.0.1:{self.ws_port}"
        deadline = asyncio.get_running_loop().time() + self.ready_timeout_seconds

        while True:
            if self._helper_process is not None and self._helper_process.returncode is not None:
                raise RuntimeError(
                    f"Telnyx WebRTC helper exited early with code {self._helper_process.returncode}",
                )

            try:
                return await self._session.ws_connect(helper_url, heartbeat=15.0)
            except aiohttp.ClientError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(f"Unable to connect to Telnyx WebRTC helper at {helper_url}") from None
                await asyncio.sleep(_HELPER_CONNECT_RETRY_SECONDS)

    async def _read_helper_messages(self) -> None:
        assert self._websocket is not None

        async for message in self._websocket:
            if message.type == aiohttp.WSMsgType.BINARY:
                await self._handle_helper_audio(message.data)
                continue

            if message.type == aiohttp.WSMsgType.TEXT:
                await self._handle_helper_control(message.data)
                continue

            if message.type in {aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                break

    async def _handle_helper_audio(self, pcm_audio: bytes) -> None:
        if not pcm_audio:
            return

        mulaw_audio = audioop.lin2ulaw(pcm_audio, PCM_SAMPLE_WIDTH)
        await self.emit_audio(mulaw_audio)

    async def _handle_helper_control(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as exc:
            self._helper_error = RuntimeError(f"Invalid control frame from Telnyx helper: {exc}")
            self._ready_event.set()
            logger.error(str(self._helper_error))
            return

        message_type = message.get("type")
        if message_type == "ready":
            logger.info(f"Telnyx WebRTC transport ready for assistant {self.assistant_id}")
            self._ready_event.set()
            return

        if message_type == "hangup":
            logger.info(f"Telnyx WebRTC helper hung up assistant {self.assistant_id}")
            self._ready_event.set()
            return

        if message_type == "error":
            error_message = str(message.get("message", "Unknown Telnyx helper error"))
            self._helper_error = RuntimeError(error_message)
            self._ready_event.set()
            logger.error(error_message)
            return

        logger.debug(f"Ignoring Telnyx helper control frame: {message}")

    async def send_audio(self, audio_data: bytes) -> None:
        if not audio_data or self._websocket is None:
            return

        pcm_audio = audioop.ulaw2lin(audio_data, PCM_SAMPLE_WIDTH)
        await self._websocket.send_bytes(pcm_audio)

    async def stop(self) -> None:
        if self._websocket is not None:
            try:
                await self._websocket.send_json({"type": "hangup"})
            except Exception:
                logger.debug(f"Failed to send helper hangup for assistant {self.assistant_id}")

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug(f"Ignoring Telnyx helper reader shutdown error: {exc}")
            self._reader_task = None

        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

        if self._session is not None:
            await self._session.close()
            self._session = None

        if self._helper_process is not None and self._helper_process.returncode is None:
            self._helper_process.terminate()
            try:
                await asyncio.wait_for(self._helper_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._helper_process.kill()
                await self._helper_process.wait()

        if self._helper_stdout_task is not None:
            await self._helper_stdout_task
            self._helper_stdout_task = None

        if self._helper_stderr_task is not None:
            await self._helper_stderr_task
            self._helper_stderr_task = None

        self._helper_process = None
        self._helper_error = None
        self._ready_event.clear()
        logger.info(f"Stopped Telnyx WebRTC transport for assistant {self.assistant_id}")
