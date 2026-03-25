"""Unit tests for the tool webhook service."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from eva.assistant.tool_webhook import ToolWebhookService


@pytest.fixture
async def webhook_service() -> ToolWebhookService:
    service = ToolWebhookService(port=9888)
    yield service
    await service.stop()


async def _make_client(service: ToolWebhookService) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=service.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestToolWebhookService:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, webhook_service: ToolWebhookService):
        async with await _make_client(webhook_service) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_executes_tool_with_direct_body(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success", "data": {"confirmation_number": "ABC123"}})
        await webhook_service.register_conversation("call-123", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/call-123/get_reservation", json={"reservation_id": "ABC123"})

        assert response.status_code == 200
        assert response.json() == {"status": "success", "data": {"confirmation_number": "ABC123"}}
        executor.execute.assert_awaited_once_with("get_reservation", {"reservation_id": "ABC123"})

    @pytest.mark.asyncio
    async def test_executes_tool_with_function_params_wrapper(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-456", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post(
                "/tools/call-456/update_reservation",
                json={"function_params": {"reservation_id": "ABC123", "seat": "12A"}},
            )

        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        executor.execute.assert_awaited_once_with(
            "update_reservation",
            {"reservation_id": "ABC123", "seat": "12A"},
        )

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_conversation(self, webhook_service: ToolWebhookService):
        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/missing/get_reservation", json={"reservation_id": "ABC123"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Unknown call_id: missing"

    @pytest.mark.asyncio
    async def test_rejects_invalid_function_params(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-789", executor)

        async with await _make_client(webhook_service) as client:
            response = await client.post("/tools/call-789/get_reservation", json={"function_params": "bad"})

        assert response.status_code == 400
        assert response.json()["detail"] == "function_params must be a JSON object"
        executor.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_unregister_conversation_removes_audit_log(self, webhook_service: ToolWebhookService):
        executor = MagicMock()
        executor.execute = AsyncMock(return_value={"status": "success"})
        await webhook_service.register_conversation("call-000", executor)
        assert await webhook_service.get_audit_log("call-000") is not None

        await webhook_service.unregister_conversation("call-000")

        assert await webhook_service.get_audit_log("call-000") is None
