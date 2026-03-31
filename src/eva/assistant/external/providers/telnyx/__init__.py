"""Telnyx provider for hosted external voice agents."""

from datetime import datetime
from typing import Any

import aiohttp
from fastapi import FastAPI, Request

from eva.assistant.external.base import BaseTelephonyTransport, ExternalAgentProvider
from eva.assistant.external.providers.telnyx.setup import TelnyxAssistantManager
from eva.assistant.external.providers.telnyx.transport import CallControlTransport, get_active_transport
from eva.models.config import TelnyxExternalAgentConfig
from eva.utils.logging import get_logger

logger = get_logger(__name__)

_TELNYX_CONVERSATIONS_API_BASE_URL = "https://api.telnyx.com/v2"


class TelnyxProvider(ExternalAgentProvider):
    """External agent provider for Telnyx AI Assistant + Call Control."""

    def __init__(self, config: TelnyxExternalAgentConfig):
        self.config = config
        self._conversation_map: dict[str, str] = {}
        self._record_ids: dict[str, str] = {}
        self._model_tag: str | None = None
        self._original_model: str | None = None

    def create_transport(self, conversation_id: str, webhook_base_url: str) -> BaseTelephonyTransport:
        return CallControlTransport(
            api_key=self.config.telnyx_api_key,
            to=self.config.sip_uri,
            app_id=self.config.call_control_app_id,
            from_number=self.config.call_control_from,
            conversation_id=conversation_id,
            webhook_base_url=webhook_base_url,
        )

    async def setup(self) -> None:
        if not (self.config.telnyx_llm and self.config.telnyx_assistant_id):
            return

        manager = TelnyxAssistantManager(api_key=self.config.telnyx_api_key)
        try:
            self._original_model = await manager.get_assistant_model(self.config.telnyx_assistant_id)
            await manager.update_assistant_model(self.config.telnyx_assistant_id, self.config.telnyx_llm)
            self._model_tag = self.config.telnyx_llm
        finally:
            await manager.close()

    async def teardown(self) -> None:
        if not (self._original_model and self.config.telnyx_assistant_id):
            return

        manager = TelnyxAssistantManager(api_key=self.config.telnyx_api_key)
        try:
            await manager.update_assistant_model(self.config.telnyx_assistant_id, self._original_model)
            logger.info("Restored assistant model to %s", self._original_model)
        except Exception:
            logger.warning("Failed to restore assistant model to %s", self._original_model, exc_info=True)
        finally:
            await manager.close()

    async def fetch_intended_speech(self, transport: BaseTelephonyTransport) -> list[dict[str, Any]]:
        conversation_id = self._resolve_conversation_id(transport)
        if not conversation_id:
            route_id = self.get_tool_call_route_id(transport)
            if route_id:
                logger.warning("No Telnyx conversation_id found for route_id %s", route_id)
            return []

        url = f"{_TELNYX_CONVERSATIONS_API_BASE_URL}/ai/conversations/{conversation_id}/messages?page[size]=100"
        timeout = aiohttp.ClientTimeout(total=30.0)

        try:
            async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.telnyx_api_key}"},
                timeout=timeout,
            ) as session:
                async with session.get(url) as response:
                    try:
                        payload = await response.json()
                    except aiohttp.ContentTypeError:
                        payload = {"message": await response.text()}

                    if response.status >= 400:
                        logger.warning(
                            "Failed to fetch Telnyx conversation messages for %s: %s %s",
                            conversation_id,
                            response.status,
                            payload,
                        )
                        return []
        except Exception as exc:
            logger.warning("Failed to fetch Telnyx conversation messages for %s: %s", conversation_id, exc)
            return []

        data = payload.get("data", [])
        if not isinstance(data, list):
            logger.warning(
                "Unexpected Telnyx conversation messages payload for %s: data was %s",
                conversation_id,
                type(data).__name__,
            )
            return []

        intended_speech: list[dict[str, Any]] = []
        for message in reversed(data):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue

            text = str(message.get("text", "")).strip()
            if not text:
                continue

            sent_at = message.get("sent_at")
            if not isinstance(sent_at, str) or not sent_at.strip():
                logger.warning("Skipping assistant message without sent_at for conversation %s", conversation_id)
                continue

            try:
                timestamp_ms = self._iso8601_to_epoch_ms(sent_at)
            except ValueError:
                logger.warning(
                    "Skipping assistant message with invalid sent_at '%s' for conversation %s",
                    sent_at,
                    conversation_id,
                )
                continue

            intended_speech.append({"text": text, "timestamp_ms": timestamp_ms})

        return intended_speech

    def register_webhook_routes(self, app: FastAPI) -> None:
        @app.post("/call-control-events")
        async def call_control_events(request: Request) -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/dynamic-variables")
        async def dynamic_variables(request: Request) -> dict[str, Any]:
            body = await request.json()
            payload = body.get("data", {}).get("payload", {})

            eva_call_id = payload.get("eva_call_id")
            conversation_id = payload.get("telnyx_conversation_id")
            call_control_id = payload.get("call_control_id")

            logger.info(
                "DV webhook: eva_call_id=%s, conversation_id=%s, cc_id=%s",
                eva_call_id,
                conversation_id,
                call_control_id,
            )

            if eva_call_id and conversation_id:
                self._conversation_map[eva_call_id] = conversation_id
                logger.info("Stored conversation mapping: %s -> %s", eva_call_id, conversation_id)

            response: dict[str, Any] = {"dynamic_variables": {}}
            if eva_call_id:
                response["dynamic_variables"]["eva_call_id"] = eva_call_id
                metadata: dict[str, str] = {"eva_call_id": eva_call_id}
                record_id = self._record_ids.get(eva_call_id)
                if record_id:
                    metadata["eva_record_id"] = record_id
                if self._model_tag:
                    metadata["eva_llm_model"] = self._model_tag
                response["conversation"] = {"metadata": metadata}
            return response

    async def get_active_transport(self, conversation_id: str) -> BaseTelephonyTransport | None:
        return await get_active_transport(conversation_id)

    def get_tool_call_route_id(self, transport: BaseTelephonyTransport) -> str | None:
        return transport.eva_call_id

    def register_record_id(self, route_id: str, record_id: str) -> None:
        self._record_ids[route_id] = record_id

    def _resolve_conversation_id(self, transport: BaseTelephonyTransport) -> str | None:
        route_id = self.get_tool_call_route_id(transport)
        if route_id:
            return self._conversation_map.get(route_id)
        return None

    @staticmethod
    def _iso8601_to_epoch_ms(value: str) -> int:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)


__all__ = [
    "CallControlTransport",
    "TelnyxAssistantManager",
    "TelnyxExternalAgentConfig",
    "TelnyxProvider",
]
