"""Factory and exports for external hosted voice agent integrations."""

from typing import overload

from eva.assistant.external.base import (
    BaseSegmentTranscriber,
    BaseTelephonyTransport,
    DeepgramSegmentTranscriber,
    ExternalAgentProvider,
    NoopSegmentTranscriber,
    create_segment_transcriber,
)
from eva.models.config import ExternalAgentConfig, TelnyxExternalAgentConfig


@overload
def get_provider(config: ExternalAgentConfig) -> ExternalAgentProvider: ...


@overload
def get_provider(config: str, *, model_config: ExternalAgentConfig) -> ExternalAgentProvider: ...


def get_provider(
    config: ExternalAgentConfig | str,
    *,
    model_config: ExternalAgentConfig | None = None,
) -> ExternalAgentProvider:
    """Create a provider instance for an external agent config."""
    if isinstance(config, str):
        provider_name = config
        if model_config is None:
            raise ValueError("model_config is required when constructing a provider by name")
    else:
        model_config = config
        provider_name = config.provider

    if provider_name == "telnyx":
        from eva.assistant.external.providers.telnyx import TelnyxProvider

        if not isinstance(model_config, TelnyxExternalAgentConfig):
            raise TypeError("Telnyx provider requires TelnyxExternalAgentConfig")
        return TelnyxProvider(model_config)

    raise ValueError(f"Unsupported external agent provider: {provider_name}")


__all__ = [
    "BaseSegmentTranscriber",
    "BaseTelephonyTransport",
    "DeepgramSegmentTranscriber",
    "ExternalAgentProvider",
    "NoopSegmentTranscriber",
    "create_segment_transcriber",
    "get_provider",
]
