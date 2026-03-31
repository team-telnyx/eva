"""FastAPI webhook service exposing EVA tools over HTTP."""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.external.base import ExternalAgentProvider
from eva.assistant.tools.tool_executor import ToolExecutor
from eva.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _ConversationRegistration:
    """Webhook routing state for a single external conversation."""

    executor: ToolExecutor
    audit_log: AuditLog = field(default_factory=AuditLog)
    on_end_call: Callable[[], Awaitable[None]] | None = None


class ToolWebhookService:
    """Expose per-conversation tool executors as HTTP webhook endpoints."""

    def __init__(
        self,
        port: int = 8888,
        host: str = "0.0.0.0",
        provider: ExternalAgentProvider | None = None,
    ):
        self.port = port
        self.host = host
        self.provider = provider

        self._lock = asyncio.Lock()
        self._conversations: dict[str, _ConversationRegistration] = {}

        self._app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._running = False

        self._register_routes()
        if self.provider is not None:
            self.provider.register_webhook_routes(self._app)

    @property
    def app(self) -> FastAPI:
        """Return the FastAPI app for in-process tests."""
        return self._app

    async def start(self) -> None:
        """Start the webhook server."""
        if self._running:
            logger.warning("Tool webhook service already running")
            return

        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        self._running = True

        while not self._server.started:
            await asyncio.sleep(0.01)

        logger.info("Tool webhook service started on http://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        """Stop the webhook server."""
        if not self._running:
            return

        self._running = False

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
        logger.info("Tool webhook service stopped")

    async def register_conversation(
        self,
        call_id: str,
        executor: ToolExecutor,
        audit_log: AuditLog | None = None,
    ) -> None:
        """Register a conversation call ID to route future webhook requests."""
        registration = _ConversationRegistration(executor=executor, audit_log=audit_log or AuditLog())
        async with self._lock:
            self._conversations[call_id] = registration
        logger.info("Registered tool webhook conversation %s", call_id)

    async def set_end_call_handler(self, call_id: str, handler: Callable[[], Awaitable[None]]) -> None:
        """Register a callback invoked when the assistant calls the end_call tool."""
        async with self._lock:
            registration = self._conversations.get(call_id)
            if registration is not None:
                registration.on_end_call = handler

    async def register_route_id(self, route_id: str, registration_key: str) -> None:
        """Register an external routing ID so tool webhook requests can be routed."""
        async with self._lock:
            registration = self._conversations.get(registration_key)
            if registration is None:
                logger.warning("Cannot register route_id %s: registration key %s not found", route_id, registration_key)
                return
            self._conversations[route_id] = registration
        logger.info("Registered route_id %s for conversation %s", route_id, registration_key)

    async def unregister_conversation(self, call_id: str) -> None:
        """Remove a conversation and all aliases from the webhook registry."""
        async with self._lock:
            registration = self._conversations.pop(call_id, None)
            if registration is not None:
                reg_id = id(registration)
                aliases = [k for k, v in self._conversations.items() if id(v) == reg_id]
                for alias in aliases:
                    del self._conversations[alias]
        logger.info("Unregistered tool webhook conversation %s", call_id)

    async def get_audit_log(self, call_id: str) -> AuditLog | None:
        """Return the audit log for a registered conversation."""
        async with self._lock:
            registration = self._conversations.get(call_id)
            return registration.audit_log if registration else None

    async def _get_registration(self, call_id: str) -> _ConversationRegistration | None:
        async with self._lock:
            return self._conversations.get(call_id)

    def _register_routes(self) -> None:
        @self._app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.post("/tools/{call_id}/{tool_name}")
        async def invoke_tool(call_id: str, tool_name: str, request: Request) -> Any:
            registration = await self._get_registration(call_id)
            if registration is None:
                from urllib.parse import unquote

                registration = await self._get_registration(unquote(call_id))
            if registration is None:
                logger.warning("Tool webhook 404: call_id=%s not found in registered conversations", call_id)
                raise HTTPException(status_code=404, detail=f"Unknown call_id: {call_id}")

            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON request body") from exc

            if body is None:
                params: dict[str, Any] = {}
            elif isinstance(body, dict):
                function_params = body.get("function_params")
                if function_params is None:
                    params = body
                elif isinstance(function_params, dict):
                    params = function_params
                else:
                    raise HTTPException(status_code=400, detail="function_params must be a JSON object")
            else:
                raise HTTPException(status_code=400, detail="Tool webhook body must be a JSON object")

            registration.audit_log.append_tool_call(tool_name, params)

            if tool_name == "end_call" and registration.on_end_call is not None:
                registration.audit_log.append_tool_response(tool_name, {"status": "call_ended"})
                logger.info("end_call tool invoked for call %s - triggering hangup", call_id)
                asyncio.create_task(registration.on_end_call())
                return {"status": "call_ended"}

            result = await registration.executor.execute(tool_name, params)
            registration.audit_log.append_tool_response(tool_name, result)

            logger.info("Executed webhook tool %s for call %s", tool_name, call_id)
            return result

        @self._app.websocket("/media-stream/{conversation_id:path}")
        async def media_stream(websocket: WebSocket, conversation_id: str) -> None:
            if self.provider is None:
                await websocket.close(code=1008, reason="No provider transport support configured")
                return

            transport = await self.provider.get_active_transport(conversation_id)
            if transport is None or not hasattr(transport, "handle_media_stream"):
                logger.warning("No active transport for media stream conversation %s", conversation_id)
                await websocket.close(code=1008, reason="No active transport for this conversation")
                return

            await websocket.accept()
            logger.info("Accepted media stream WebSocket for conversation %s", conversation_id)
            try:
                await transport.handle_media_stream(websocket)
            except WebSocketDisconnect:
                logger.info("Media stream WebSocket disconnected for %s", conversation_id)
            except Exception as exc:
                logger.error("Media stream error for %s: %s", conversation_id, exc)
