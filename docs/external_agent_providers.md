# External Agent Providers

Benchmark any hosted voice agent API as a black box using EVA's external agent provider pattern. The bridge connects EVA's ElevenLabs user simulator to an external voice assistant via a pluggable transport, while routing the assistant's tool calls through EVA's deterministic scenario database.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│  EVA Runner                                                           │
│                                                                       │
│  ┌──────────────┐     ┌──────────────────────┐     ┌───────────────┐ │
│  │ ElevenLabs   │ WS  │  External Agent       │     │ Tool Webhook  │ │
│  │ User Sim     │◄───►│  Bridge (generic)     │     │ Service       │ │
│  │              │     │                       │     │ :8888         │ │
│  └──────────────┘     └──────────┬───────────┘     └───────┬───────┘ │
│                                  │                         │         │
│                    Provider      │ Audio Stream             │ Public  │
│                    Transport     │                          │ URL     │
└──────────────────────────────────┼─────────────────────────┼─────────┘
                                   │                         │
                          ┌────────▼─────────┐     ┌────────▼────────┐
                          │  Provider         │     │  External       │
                          │  Platform         │────►│  Voice Agent    │
                          └──────────────────┘     └─────────────────┘
```

## How It Works

1. The **provider transport** connects to the external voice agent (e.g., via Call Control API, WebSocket, REST)
2. Audio flows bidirectionally between the transport and the user simulator
3. The assistant's **tool calls** hit the tool webhook service, which executes them against EVA's deterministic scenario database
4. EVA records transcripts, audio, tool calls, and metrics — identical output format to Pipecat evaluations

## Why a Client-Side Bridge?

The bridge sits between the user simulator and the external agent, which adds a hop that doesn't exist in production calls. This is intentional:

1. **Metrics compatibility.** EVA's metrics pipeline expects `pipecat_logs.jsonl` format with precise audio timestamps, VAD events, and turn boundaries. The bridge captures these from the raw audio stream — something the external agent's API doesn't expose.

2. **Platform-independent latency measurement.** With the bridge observing both sides of the audio stream, latency measurements (time-to-first-byte, response latency) are taken from a neutral third-party perspective, not self-reported by the provider. This makes cross-provider comparisons fair.

**Important caveat:** Real production calls between a user and an external voice agent do *not* have this intermediary. The bridge adds network hops and processing overhead, so **latency numbers from EVA benchmarks will be higher than real-world latency** for the same provider. EVA measures relative performance across providers under identical conditions — not absolute production latency.

## Adding a New Provider

Implement `ExternalAgentProvider` (in `src/eva/assistant/external/base.py`):

```python
class ExternalAgentProvider(ABC):
    """Plugin interface for hosted voice agent APIs."""

    @abstractmethod
    def create_transport(self, conversation_id: str, webhook_base_url: str) -> BaseTelephonyTransport:
        """Create a transport connection to the external agent."""

    @abstractmethod
    async def setup(self) -> None:
        """One-time setup before a benchmark run (e.g., configure assistant via API)."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up after a benchmark run (e.g., restore original config)."""

    @abstractmethod
    async def fetch_intended_speech(self, transport: BaseTelephonyTransport) -> list[dict]:
        """Fetch intended assistant speech for metrics (post-call enrichment)."""

    def register_webhook_routes(self, app: FastAPI) -> None:
        """Register provider-specific webhook routes. Default: no-op."""

    async def get_active_transport(self, conversation_id: str) -> BaseTelephonyTransport | None:
        """Look up active transport by conversation ID (for media stream routing)."""

    def get_tool_call_route_id(self, transport: BaseTelephonyTransport) -> str | None:
        """Return the ID the external agent uses in tool webhook URLs."""

    def register_record_id(self, route_id: str, record_id: str) -> None:
        """Associate a routing ID with an EVA record for metadata tagging."""
```

Then create a provider directory:

```
src/eva/assistant/external/providers/yourprovider/
├── __init__.py      # YourProvider(ExternalAgentProvider)
├── transport.py     # Transport implementation (BaseTelephonyTransport)
└── setup.py         # Optional: API client for agent configuration
```

Register it in `src/eva/assistant/external/__init__.py`:

```python
def get_provider(model_config: ExternalAgentConfig) -> ExternalAgentProvider:
    if provider_name == "yourprovider":
        from .providers.yourprovider import YourProvider
        return YourProvider(model_config)
```

Add a config subclass in `src/eva/models/config.py`:

```python
class YourProviderConfig(ExternalAgentConfig):
    provider: Literal["yourprovider"] = "yourprovider"
    # provider-specific fields...
```

**That's it.** No changes needed to the bridge, worker, runner, or metrics pipeline.

## Transport Layer

The architecture has two independent abstraction layers that can be mixed and matched:

### `BaseTelephonyTransport` — Audio transport

The low-level ABC that handles bidirectional audio streaming with the external agent:

```python
class BaseTelephonyTransport(ABC):
    async def start(self) -> None: ...       # Open the connection
    async def stop(self) -> None: ...        # Close the connection
    async def send_audio(self, data: bytes) -> None: ...  # Send PCM audio to agent
    async def emit_audio(self, data: bytes) -> None: ...  # Receive audio from agent
```

Different agents connect differently — some via SIP/Call Control (place a phone call), some via WebSocket (connect to an API endpoint), some via WebRTC. Each connectivity method is a different transport implementation.

### `ExternalAgentProvider` — Provider lifecycle

The higher-level ABC that owns the transport and manages the agent's configuration:

- Creates transport instances via `create_transport()`
- Handles run-level setup/teardown (e.g., configuring the assistant's system prompt and tools via API)
- Fetches post-call data for metrics enrichment

### Mix-and-match

These layers are **independent**. A provider creates transports, but a transport doesn't know about its provider. This means:

- **Different providers can share a transport.** Any voice agent reachable via SIP could reuse the Telnyx Call Control transport — you'd just need a different provider to handle setup/teardown and post-call enrichment.
- **One provider could offer multiple transports.** A provider that supports both SIP and WebSocket connectivity could let users choose.
- **Transports are simple to implement.** A WebSocket-based transport (e.g., for ElevenLabs Conversational AI) just needs `start()`, `stop()`, and `send_audio()` — no telephony knowledge required.

## Telnyx Provider

The included Telnyx provider (`providers/telnyx/`) serves as the reference implementation.

### Transport: Call Control

The Telnyx transport (`CallControlTransport`) works by:

1. **Placing an outbound call** via the [Telnyx Call Control API](https://developers.telnyx.com/docs/api/v2/call-control) to the assistant's SIP URI
2. **Receiving a bidirectional media stream** over WebSocket — the assistant's audio comes in, EVA's user simulator audio goes out
3. **Routing tool calls** via a dynamic variable (`eva_call_id`) injected as a SIP header, which the assistant includes in webhook URLs for deterministic routing to the correct EVA scenario database

Audio format is **L16 16kHz PCM** (the Telnyx Call Control media stream default). The bridge handles sample rate conversion to EVA's 24kHz recording format.

This transport would work for any voice agent reachable via SIP — not just Telnyx AI Assistants.

### Prerequisites

- Telnyx account with a **Call Control Application**, **phone number**, and **API key** (v2)
- A **stable ngrok domain** for tool webhooks
- ElevenLabs API key + Conversational AI agent(s) for user simulation

### Configuration

```bash
# Provider selection (auto-detected from sip_uri if omitted)
export EVA_MODEL__PROVIDER="telnyx"

# Telnyx Call Control
export EVA_MODEL__SIP_URI="sip:assistant@sip.telnyx.com"
export EVA_MODEL__TELNYX_API_KEY="KEY..."
export EVA_MODEL__CALL_CONTROL_APP_ID="1234567890"
export EVA_MODEL__CALL_CONTROL_FROM="+12125551234"

# Tool webhooks via ngrok
export EVA_MODEL__WEBHOOK_BASE_URL="https://your-domain.ngrok-free.dev"

# ElevenLabs user simulator
export ELEVENLABS_API_KEY="..."
export ELEVENLABS_USER_AGENT_ID_USER_PERSONA_1="agent_..."

# LLM judge (for metrics)
export EVA_MODEL_LIST='[{"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o", "api_key": "sk-..."}}]'

# Optional: STT for transcript generation
export EVA_MODEL__STT="deepgram"
export EVA_MODEL__STT_PARAMS='{"model": "nova-3"}'
```

### Running

```bash
# Start ngrok (separate terminal)
ngrok http 8888 --url your-domain.ngrok-free.dev

# Run evaluation
python -m eva --record-ids 1.1.2
```

### How Telnyx routing works

Tool webhook URLs use `{{eva_call_id}}` — a dynamic variable injected via the `X-Eva-Call-Id` SIP header. EVA generates this ID before dialing, enabling deterministic routing for concurrent benchmark calls.

### Post-call enrichment

After each call, the Telnyx provider fetches the full conversation history from the Conversations API, enriching the local audit log with user/assistant speech turns and tool call details.

## Changes to Core EVA

This section documents every edit made to existing upstream files and the rationale.

### User simulator (`user_simulator/client.py`, `audio_interface.py`)

**What changed:** Added `audio_codec` parameter ("mulaw" or "pcm") and `events_output_path` for event log location.

**Why:** External agents may use different audio formats. Pipecat uses 8kHz μ-law (Twilio convention); Call Control-based transports use 16kHz L16 PCM. The codec parameter lets the user simulator match whatever the transport expects. Zero behavior change when using the default `"mulaw"`.

### Config (`models/config.py`)

**What changed:** Added `ExternalAgentConfig` base class alongside existing `PipelineConfig`, `SpeechToSpeechConfig`, `AudioLLMConfig`. Updated the discriminator to route `provider` or `sip_uri` fields to the external agent config.

**Why:** Same pattern as the existing config hierarchy — each evaluation mode has its own config class. The discriminator auto-detects the mode from env vars.

### Orchestrator (`orchestrator/runner.py`, `worker.py`)

**What changed:** Runner creates and manages provider lifecycle (setup/teardown). Worker instantiates `ExternalAgentBridgeServer` when config is `ExternalAgentConfig`.

**Why:** Mirrors how the runner already manages `AssistantServer` for Pipecat evaluations. The `if is_external_agent:` branches parallel the existing pipeline logic.

### Metrics base (`metrics/base.py`)

**What changed:** Added `message_trace` field to `MetricContext`, plus `get_transcript_trace()` and `get_num_turns()` helper methods on judge base classes.

**Why:** External agents produce two trace types: a *spoken* trace (what was actually heard, post-interruption/truncation) and a *message* trace (full intended messages). Some metrics should evaluate spoken output; others should evaluate message-level quality. The helpers let individual metrics choose which trace to use. Default is `conversation_trace` (no behavior change for existing metrics).

### Metrics processor (`metrics/processor.py`)

**What changed:**
- Control event filtering: skip `<call:end_call>` pseudo-speech events
- Event sort priority: process audio boundary events before speech at same timestamp
- Empty content guard: skip audit-log assistant entries with no content
- Per-entry `_skip_truncation` flag: entries marked `already_spoken` (by the bridge's metrics adapter) skip spoken-prefix truncation
- Load `message_trace.jsonl` when present; fall back to `conversation_trace` copy
- Removed `_build_message_trace`, `_reconcile_message_trace`, and `is_bridge` detection

**Why:** The bridge writes its audit log with `already_spoken: true` on assistant entries (since they come from STT, not raw LLM output), and pre-builds `message_trace.jsonl`. The processor just reads these files — no bridge-mode branching needed. The control event filter and sort priority are general correctness improvements that benefit all evaluation modes.

### Individual metrics (faithfulness, conciseness, conversation_progression, speakability, authentication_success)

**What changed:** Use `self.get_num_turns(context)` instead of `len(context.conversation_trace)`. Speakability strips annotation labels before judging. Authentication deserializes JSON-string tool responses.

**Why:** Consistent use of base class helpers. The annotation stripping and JSON deserialization are correctness fixes independent of the external agent pattern.

### Tools (`assistant/tools/airline_tools.py`)

**What changed:** Added `end_call` tool function.

**Why:** External voice agents use an `end_call` tool to hang up. The tool webhook service intercepts this to trigger transport disconnect, but the tool executor still needs a registered handler.

### Audit log (`assistant/agentic/audit_log.py`)

**What changed:** Added `replace_transcript()` method.

**Why:** The bridge fetches the full conversation from the provider API post-call and replaces the local tool-only transcript. Generic method — not provider-specific.

## Audio Format

- **Transport audio**: Depends on provider (Telnyx: L16 16kHz; others may differ)
- **User simulator output**: Configurable via `audio_codec` parameter
- **Recordings**: PCM 16-bit WAV at 24kHz (EVA standard)
- The bridge handles sample rate conversion between the transport and EVA's recording format

## Stable Webhook URL (Required)

Ephemeral ngrok URLs change on restart, causing webhook mismatches. EVA rejects them:

1. Go to https://dashboard.ngrok.com/domains
2. Claim a free static domain (e.g., `your-name.ngrok-free.dev`)
3. Start ngrok: `ngrok http 8888 --url your-name.ngrok-free.dev`
