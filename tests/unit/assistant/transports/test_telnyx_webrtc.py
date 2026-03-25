"""Unit tests for the Telnyx WebRTC transport."""

import asyncio
import json
from pathlib import Path

import pytest

try:
    import audioop
except ImportError:
    import audioop_lts as audioop

from eva.assistant.transports.telnyx_webrtc import TelnyxWebRTCTransport


def _write_helper(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


class TestTelnyxWebRTCTransport:
    @pytest.mark.asyncio
    async def test_start_send_audio_and_stop(self, tmp_path: Path) -> None:
        helper_path = tmp_path / "mock_helper.js"
        _write_helper(
            helper_path,
            """\
import asyncio
import json
import sys

from websockets.server import serve


def parse_args(argv):
    parsed = {}
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and index + 1 < len(argv):
            parsed[token[2:]] = argv[index + 1]
            index += 2
            continue
        index += 1
    return parsed


async def main():
    args = parse_args(sys.argv)
    port = int(args["ws-port"])

    async def handler(websocket):
        await websocket.send(json.dumps({"type": "ready"}))
        await websocket.send(bytes([0x00, 0x00, 0xFF, 0x7F]))
        async for message in websocket:
            if isinstance(message, bytes):
                await websocket.send(message)
                continue
            payload = json.loads(message)
            if payload["type"] == "hangup":
                await websocket.close()
                break

    async with serve(handler, "127.0.0.1", port):
        await asyncio.Future()


asyncio.run(main())
""",
        )

        received_audio: list[bytes] = []
        transport = TelnyxWebRTCTransport(
            assistant_id="assistant-123",
            conversation_id="conversation-123",
            webhook_base_url="https://example.com",
            node_executable="python3",
            helper_script_path=helper_path,
            install_dependencies=False,
        )
        transport.set_audio_handler(lambda audio: _append_audio(received_audio, audio))

        await transport.start()

        inbound_pcm = bytes([0x00, 0x00, 0xFF, 0x7F])
        expected_inbound_mulaw = audioop.lin2ulaw(inbound_pcm, 2)
        await _wait_for(lambda: len(received_audio) >= 1)
        assert received_audio[0] == expected_inbound_mulaw

        user_mulaw = audioop.lin2ulaw(bytes([0x00, 0x00, 0x34, 0x12]), 2)
        await transport.send_audio(user_mulaw)
        await _wait_for(lambda: len(received_audio) >= 2)
        assert received_audio[1] == user_mulaw

        await transport.stop()

    @pytest.mark.asyncio
    async def test_start_raises_when_helper_reports_error(self, tmp_path: Path) -> None:
        helper_path = tmp_path / "error_helper.js"
        _write_helper(
            helper_path,
            """\
import asyncio
import json
import sys

from websockets.server import serve


def parse_args(argv):
    parsed = {}
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and index + 1 < len(argv):
            parsed[token[2:]] = argv[index + 1]
            index += 2
            continue
        index += 1
    return parsed


async def main():
    args = parse_args(sys.argv)
    port = int(args["ws-port"])

    async def handler(websocket):
        await websocket.send(json.dumps({"type": "error", "message": "helper exploded"}))
        await websocket.close()

    async with serve(handler, "127.0.0.1", port):
        await asyncio.Future()


asyncio.run(main())
""",
        )

        transport = TelnyxWebRTCTransport(
            assistant_id="assistant-123",
            conversation_id="conversation-123",
            webhook_base_url="https://example.com",
            node_executable="python3",
            helper_script_path=helper_path,
            install_dependencies=False,
        )

        with pytest.raises(RuntimeError, match="helper exploded"):
            await transport.start()

        await transport.stop()


async def _append_audio(received_audio: list[bytes], audio: bytes) -> None:
    received_audio.append(audio)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await asyncio.sleep(0.01)
