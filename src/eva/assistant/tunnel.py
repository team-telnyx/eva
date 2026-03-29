"""Cloudflare Quick Tunnel management for local webhook exposure."""

import asyncio
import contextlib
import re

from eva.utils.logging import get_logger

logger = get_logger(__name__)

_TRY_CLOUDFLARE_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_STARTUP_TIMEOUT_SECONDS = 15.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class CloudflareTunnel:
    """Manage a `cloudflared tunnel` subprocess for a local HTTP service."""

    def __init__(
        self,
        port: int,
        *,
        binary: str = "cloudflared",
        startup_timeout_seconds: float = _STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.port = port
        self.binary = binary
        self.startup_timeout_seconds = startup_timeout_seconds

        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._url_ready = asyncio.Event()
        self._url: str | None = None
        self._stderr_lines: list[str] = []

    @property
    def url(self) -> str:
        """Return the discovered public tunnel URL."""
        if self._url is None:
            raise RuntimeError("Cloudflare tunnel URL is not available")
        return self._url

    async def __aenter__(self) -> "CloudflareTunnel":
        if self._process is not None:
            raise RuntimeError("Cloudflare tunnel already started")

        self._process = await asyncio.create_subprocess_exec(
            self.binary,
            "tunnel",
            "--url",
            f"http://localhost:{self.port}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._consume_stderr())

        try:
            await asyncio.wait_for(self._wait_for_url(), timeout=self.startup_timeout_seconds)
        except Exception:
            await self._shutdown()
            raise

        logger.info("Cloudflare tunnel URL: %s", self.url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._shutdown()

    async def _wait_for_url(self) -> None:
        if self._process is None:
            raise RuntimeError("Cloudflare tunnel process was not started")

        url_ready_task = asyncio.create_task(self._url_ready.wait())
        process_wait_task = asyncio.create_task(self._process.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {url_ready_task, process_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if url_ready_task in done:
                    return
                if process_wait_task in done:
                    stderr_tail = "\n".join(self._stderr_lines[-10:])
                    raise RuntimeError(
                        "cloudflared exited before publishing a tunnel URL"
                        + (f":\n{stderr_tail}" if stderr_tail else "")
                    )
        finally:
            url_ready_task.cancel()
            process_wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await url_ready_task
            with contextlib.suppress(asyncio.CancelledError):
                await process_wait_task

    async def _consume_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        while True:
            line = await process.stderr.readline()
            if not line:
                return

            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue

            self._stderr_lines.append(decoded)
            if len(self._stderr_lines) > 50:
                self._stderr_lines = self._stderr_lines[-50:]

            if self._url is None and (match := _TRY_CLOUDFLARE_URL_RE.search(decoded)):
                self._url = match.group(0)
                self._url_ready.set()

    async def _shutdown(self) -> None:
        process = self._process
        self._process = None

        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if self._stderr_task is not None:
            if not self._stderr_task.done():
                self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
