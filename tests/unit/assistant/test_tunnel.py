"""Unit tests for Cloudflare Quick Tunnel management."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from eva.assistant.tunnel import CloudflareTunnel


class _FakeStderr:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return next(self._lines, b"")


class _FakeProcess:
    def __init__(self, stderr_lines: list[bytes]) -> None:
        self.stderr = _FakeStderr(stderr_lines)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._wait_event = asyncio.Event()

    async def wait(self) -> int:
        await self._wait_event.wait()
        return self.returncode or 0

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self._wait_event.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._wait_event.set()


class TestCloudflareTunnel:
    @pytest.mark.asyncio
    async def test_discovers_trycloudflare_url_and_terminates_process(self, monkeypatch):
        process = _FakeProcess(
            [
                b"INF Initial protocol quic\n",
                b"INF Your quick Tunnel has been created! Visit it at https://eva-demo.trycloudflare.com\n",
            ]
        )

        monkeypatch.setattr(
            "eva.assistant.tunnel.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        )

        async with CloudflareTunnel(port=8888) as tunnel:
            assert tunnel.url == "https://eva-demo.trycloudflare.com"

        assert process.terminated is True
        assert process.killed is False

    @pytest.mark.asyncio
    async def test_times_out_when_cloudflared_never_emits_url(self, monkeypatch):
        process = _FakeProcess([b"INF waiting for connection\n"])

        monkeypatch.setattr(
            "eva.assistant.tunnel.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        )

        tunnel = CloudflareTunnel(port=8888, startup_timeout_seconds=0.01)

        with pytest.raises(asyncio.TimeoutError):
            await tunnel.__aenter__()

        assert process.terminated is True
