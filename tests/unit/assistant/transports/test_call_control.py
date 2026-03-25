"""Unit tests for the Telnyx Call Control transport."""

import asyncio
import base64
import json

from aiohttp import web
import pytest
from websockets.asyncio.client import connect

from eva.assistant.transports.call_control import CallControlTransport


class TestCallControlTransport:
    @pytest.mark.asyncio
    async def test_start_send_audio_and_stop(self, unused_tcp_port: int) -> None:
        requests: list[dict] = []
        create_call_received = asyncio.Event()
        api_runner = await _start_api_server(unused_tcp_port, requests, create_call_received)

        received_audio: list[bytes] = []
        transport = CallControlTransport(
            api_key="telnyx-key",
            to="sip:test@example.com",
            stream_url="wss://stream.example.com/media",
            connection_id="connection-123",
            from_number="+15551234567",
            conversation_id="conversation-123",
            webhook_base_url="https://example.com",
            api_base_url=f"http://127.0.0.1:{unused_tcp_port}/v2",
        )
        transport.set_audio_handler(lambda audio: _append_audio(received_audio, audio))

        try:
            start_task = asyncio.create_task(transport.start())
            await asyncio.wait_for(create_call_received.wait(), timeout=5.0)

            assert requests[0]["path"] == "/v2/calls"
            assert requests[0]["json"] == {
                "connection_id": "connection-123",
                "to": "sip:test@example.com",
                "from": "+15551234567",
                "stream_url": "wss://stream.example.com/media",
                "stream_track": "both_tracks",
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": "PCMU",
            }

            async with connect(f"ws://127.0.0.1:{transport.local_ws_port}") as websocket:
                await start_task

                await websocket.send(
                    json.dumps(
                        {
                            "event": "start",
                            "stream_id": "stream-123",
                            "start": {
                                "call_control_id": "call-control-123",
                                "media_format": {"encoding": "PCMU", "sample_rate": 8000},
                            },
                        }
                    )
                )

                inbound_audio = b"\xff" * 160
                await websocket.send(
                    json.dumps(
                        {
                            "event": "media",
                            "media": {
                                "track": "inbound",
                                "payload": base64.b64encode(inbound_audio).decode("ascii"),
                            },
                        }
                    )
                )
                await _wait_for(lambda: received_audio == [inbound_audio])

                outbound_audio = b"\x7f" * 160
                await transport.send_audio(outbound_audio)
                outbound_message = json.loads(await websocket.recv())
                assert outbound_message == {
                    "event": "media",
                    "media": {"payload": base64.b64encode(outbound_audio).decode("ascii")},
                }

                await transport.stop()

            assert requests[1]["path"] == "/v2/calls/call-control-123/actions/hangup"
            assert requests[1]["json"] == {}
        finally:
            await transport.stop()
            await api_runner.cleanup()

    @pytest.mark.asyncio
    async def test_start_raises_when_call_creation_fails(self, unused_tcp_port: int) -> None:
        async def failing_create_call(_request: web.Request) -> web.Response:
            return web.json_response({"errors": [{"detail": "boom"}]}, status=500)

        app = web.Application()
        app.router.add_post("/v2/calls", failing_create_call)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
        await site.start()

        transport = CallControlTransport(
            api_key="telnyx-key",
            to="sip:test@example.com",
            stream_url="wss://stream.example.com/media",
            connection_id="connection-123",
            from_number="+15551234567",
            conversation_id="conversation-123",
            webhook_base_url="https://example.com",
            api_base_url=f"http://127.0.0.1:{unused_tcp_port}/v2",
        )

        try:
            with pytest.raises(RuntimeError, match="status 500"):
                await transport.start()

            assert transport._server is None
            assert transport._session is None
        finally:
            await transport.stop()
            await runner.cleanup()


async def _start_api_server(port: int, requests: list[dict], create_call_received: asyncio.Event) -> web.AppRunner:
    async def create_call(request: web.Request) -> web.Response:
        requests.append(
            {
                "path": request.path,
                "json": await request.json(),
            }
        )
        create_call_received.set()
        return web.json_response({"data": {"call_control_id": "call-control-123"}})

    async def hangup(request: web.Request) -> web.Response:
        requests.append(
            {
                "path": request.path,
                "json": await request.json(),
            }
        )
        return web.json_response({"data": {"result": "ok"}})

    app = web.Application()
    app.router.add_post("/v2/calls", create_call)
    app.router.add_post("/v2/calls/{call_control_id}/actions/hangup", hangup)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def _append_audio(received_audio: list[bytes], audio: bytes) -> None:
    received_audio.append(audio)


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for condition")
        await asyncio.sleep(0.01)
