"""Telnyx AI Assistant setup utilities for EVA benchmarking."""

import json
from pathlib import Path
from typing import Any

import aiohttp

from eva.models.agents import AgentConfig, AgentTool
from eva.utils.logging import get_logger

logger = get_logger(__name__)


class TelnyxAssistantManager:
    """Create and manage Telnyx AI assistants for benchmark runs."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.telnyx.com",
        session: aiohttp.ClientSession | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self._session_owner = session is None
        self.session = session or aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30.0),
        )

    # Telnyx platform defaults (match portal "Default Assistant" template)
    DEFAULT_MODEL = "moonshotai/Kimi-K2.5"
    DEFAULT_VOICE = "Telnyx.Ultra.a7a59115-2425-4192-844c-1e98ec7d6877"
    DEFAULT_STT_MODEL = "deepgram/flux"

    async def create_benchmark_assistant(
        self,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str,
        model: str | None = None,
        voice: str | None = None,
    ) -> str:
        """Create a Telnyx assistant configured to call EVA tool webhooks."""
        assistant_payload = self._build_assistant_payload(
            agent_config=agent_config,
            agent_config_path=agent_config_path,
            webhook_base_url=webhook_base_url,
            model=model or self.DEFAULT_MODEL,
            voice=voice or self.DEFAULT_VOICE,
        )

        url = f"{self.api_base}/v2/ai/assistants"
        logger.info(f"Creating Telnyx benchmark assistant for agent {agent_config.id}")

        async with self.session.post(url, json=assistant_payload) as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to create Telnyx assistant: {response.status} {json.dumps(payload, sort_keys=True)}",
                )

        assistant_id = str(payload.get("id", "")).strip()
        if not assistant_id:
            raise RuntimeError(f"Telnyx assistant creation response did not include an id: {payload}")

        logger.info(f"Created Telnyx benchmark assistant {assistant_id} for agent {agent_config.id}")
        return assistant_id

    async def setup_assistant(
        self,
        assistant_id: str,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str,
        model: str | None = None,
    ) -> None:
        """PATCH an existing assistant with EVA webhook URLs and optional model."""
        await self._setup_or_patch_assistant(
            assistant_id=assistant_id,
            agent_config=agent_config,
            agent_config_path=agent_config_path,
            webhook_base_url=webhook_base_url,
            model=model,
        )

    async def delete_assistant(self, assistant_id: str) -> None:
        """Delete a Telnyx benchmark assistant."""
        url = f"{self.api_base}/v2/ai/assistants/{assistant_id}"
        logger.info(f"Deleting Telnyx benchmark assistant {assistant_id}")

        async with self.session.delete(url) as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to delete Telnyx assistant {assistant_id}: "
                    f"{response.status} {json.dumps(payload, sort_keys=True)}",
                )

        logger.info(f"Deleted Telnyx benchmark assistant {assistant_id}")

    async def get_assistant_model(self, assistant_id: str) -> str:
        """Get the current LLM model for a Telnyx assistant."""
        url = f"{self.api_base}/v2/ai/assistants/{assistant_id}"
        async with self.session.get(url) as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to get Telnyx assistant {assistant_id}: "
                    f"{response.status} {json.dumps(payload, sort_keys=True)}",
                )
            # API may return model at top level or under "data"
            model = payload.get("model") or payload.get("data", {}).get("model", "")
            logger.info(f"Current model for assistant {assistant_id}: {model}")
            return model

    async def update_assistant_model(self, assistant_id: str, model: str) -> None:
        """PATCH the LLM model on an existing Telnyx assistant."""
        url = f"{self.api_base}/v2/ai/assistants/{assistant_id}"
        logger.info(f"Updating assistant {assistant_id} model to {model}")
        async with self.session.patch(url, json={"model": model}) as response:
            payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to update assistant model: "
                    f"{response.status} {json.dumps(payload, sort_keys=True)}",
                )
        logger.info(f"Updated assistant {assistant_id} model to {model}")

    async def _setup_or_patch_assistant(
        self,
        assistant_id: str,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str,
        model: str | None = None,
    ) -> None:
        """Update an existing assistant so its webhooks point at the current EVA tunnel.

        Fetches the current assistant config and replaces webhook base URLs in-place
        rather than rebuilding the full tool payload from the agent config. This avoids
        Telnyx validation issues that can arise from sending a fully reconstructed payload.
        """
        normalized_webhook_base = webhook_base_url.rstrip("/")
        get_url = f"{self.api_base}/v2/ai/assistants/{assistant_id}"

        # Fetch current assistant config
        async with self.session.get(get_url) as response:
            current = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to fetch assistant {assistant_id}: "
                    f"{response.status} {json.dumps(current, sort_keys=True)}",
                )

        # Find the current webhook base URL from existing tool URLs or DV webhook
        old_base = None
        dv_url = current.get("dynamic_variables_webhook_url", "")
        if dv_url:
            # Extract base URL: everything before /dynamic-variables
            idx = dv_url.find("/dynamic-variables")
            if idx > 0:
                old_base = dv_url[:idx]
        if not old_base:
            for tool in current.get("tools", []):
                tool_url = tool.get("webhook", {}).get("url", "")
                if "/tools/" in tool_url:
                    old_base = tool_url.split("/tools/")[0]
                    break

        if not old_base:
            logger.warning(
                "Could not detect existing webhook base URL on assistant %s; "
                "falling back to full payload rebuild",
                assistant_id,
            )
            payload = self._build_assistant_update_payload(
                agent_config=agent_config,
                agent_config_path=agent_config_path,
                webhook_base_url=normalized_webhook_base,
            )
        else:
            # Replace old base URL with new one in existing config
            payload: dict[str, Any] = {}
            if dv_url:
                payload["dynamic_variables_webhook_url"] = dv_url.replace(old_base, normalized_webhook_base)

            updated_tools = []
            for tool in current.get("tools", []):
                tool_copy = json.loads(json.dumps(tool))  # deep copy
                wh = tool_copy.get("webhook", {})
                if "url" in wh:
                    wh["url"] = wh["url"].replace(old_base, normalized_webhook_base)
                updated_tools.append(tool_copy)
            if updated_tools:
                payload["tools"] = updated_tools

        if model:
            payload["model"] = model

        logger.info(
            "Patching Telnyx assistant %s: webhook base %s → %s%s",
            assistant_id,
            old_base or "(unknown)",
            normalized_webhook_base,
            f", model → {model}" if model else "",
        )
        async with self.session.patch(get_url, json=payload) as response:
            response_payload = await self._parse_response_json(response)
            if response.status >= 400:
                raise RuntimeError(
                    f"Failed to patch Telnyx assistant {assistant_id}: "
                    f"{response.status} {json.dumps(response_payload, sort_keys=True)}",
                )
        logger.info("Patched Telnyx assistant %s webhook URLs", assistant_id)

    async def close(self) -> None:
        """Close the underlying HTTP session if owned by the manager."""
        if self._session_owner and not self.session.closed:
            await self.session.close()

    @staticmethod
    async def _parse_response_json(response: aiohttp.ClientResponse) -> dict[str, Any]:
        try:
            payload = await response.json()
        except aiohttp.ContentTypeError:
            payload = {"message": await response.text()}
        return payload if isinstance(payload, dict) else {"data": payload}

    def _build_assistant_payload(
        self,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str,
        model: str,
        voice: str,
    ) -> dict[str, Any]:
        normalized_webhook_base = webhook_base_url.rstrip("/")

        # Build tools: EVA scenario webhook tools + default hangup
        tools: list[dict[str, Any]] = [
            self._build_webhook_tool(tool, normalized_webhook_base)
            for tool in agent_config.tools
        ]
        # If the agent config includes an end_call tool, it will be registered
        # as a webhook tool and will trigger hangup via our Call Control API.
        # Only add the built-in hangup tool if no end_call webhook tool exists.
        has_end_call_tool = any(t.id == "end_call" for t in agent_config.tools)
        if not has_end_call_tool:
            tools.append({
                "type": "hangup",
                "hangup": {
                    "description": (
                        "To be used whenever the conversation has ended "
                        "and it would be appropriate to hangup the call."
                    ),
                },
            })

        return {
            "name": f"EVA Benchmark - {agent_config.name}",
            "description": (
                f"Auto-generated by EVA from {Path(agent_config_path).name} "
                f"for benchmark agent {agent_config.id}."
            ),
            "instructions": self._build_system_prompt(agent_config),
            "model": model,
            "greeting": agent_config.greeting if hasattr(agent_config, "greeting") and agent_config.greeting else "Hello, how can I help you today?",
            "tools": tools,
            "enabled_features": ["telephony"],
            "voice_settings": {
                "voice": voice,
                "voice_speed": 1.2,
                "expressive_mode": False,
                "language_boost": "English",
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True,
                "background_audio": {
                    "type": "predefined_media",
                    "value": "silence",
                    "volume": 0.5,
                },
            },
            "transcription": {
                "model": self.DEFAULT_STT_MODEL,
                "language": "en",
                "settings": {
                    "eot_threshold": 0.9,
                    "eot_timeout_ms": 5000,
                    "eager_eot_threshold": 0.9,
                },
            },
            "telephony_settings": {
                "supports_unauthenticated_web_calls": True,
                "time_limit_secs": 600,
                "user_idle_timeout_secs": 60,
            },
            "interruption_settings": {
                "enable": True,
                "start_speaking_plan": {
                    "wait_seconds": 0.1,
                    "transcription_endpointing_plan": {
                        "on_punctuation_seconds": 0.1,
                        "on_no_punctuation_seconds": 0.1,
                        "on_number_seconds": 0.1,
                    },
                },
            },
            "privacy_settings": {
                "data_retention": True,
                "pii_redaction": "disabled",
            },
            "dynamic_variables_webhook_url": f"{normalized_webhook_base}/dynamic-variables",
            "dynamic_variables": {
                "eva_call_id": None,
            },
        }

    def _build_assistant_update_payload(
        self,
        agent_config: AgentConfig,
        agent_config_path: str,
        webhook_base_url: str,
    ) -> dict[str, Any]:
        """Build the assistant PATCH payload for runtime webhook updates."""
        payload = self._build_assistant_payload(
            agent_config=agent_config,
            agent_config_path=agent_config_path,
            webhook_base_url=webhook_base_url,
            model=self.DEFAULT_MODEL,
            voice=self.DEFAULT_VOICE,
        )
        return {
            "tools": payload["tools"],
            "dynamic_variables_webhook_url": payload["dynamic_variables_webhook_url"],
            "dynamic_variables": payload["dynamic_variables"],
        }

    @staticmethod
    def _build_system_prompt(agent_config: AgentConfig) -> str:
        sections = [
            agent_config.description.strip(),
            f"Role: {agent_config.role.strip()}",
            agent_config.personality.strip() if agent_config.personality else "",
            agent_config.instructions.strip(),
        ]
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _build_webhook_tool(tool: AgentTool, webhook_base_url: str) -> dict[str, Any]:
        return {
            "type": "webhook",
            "timeout_ms": 5000,
            "webhook": {
                "name": tool.id,
                "description": tool.description,
                "url": f"{webhook_base_url}/tools/{{{{eva_call_id}}}}/{tool.id}",
                "method": "POST",
                "path_parameters": {"type": "object", "properties": {}},
                "query_parameters": {"type": "object", "properties": {}},
                "body_parameters": {
                    "type": "object",
                    "properties": tool.get_parameter_properties(),
                    "required": tool.get_required_param_names(),
                },
                "headers": [],
            },
        }
