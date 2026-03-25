# Telephony Bridge — Benchmarking External Voice Assistants

Benchmark any voice assistant as a black box using EVA's telephony bridge. The bridge connects EVA's ElevenLabs user simulator to an external assistant via WebRTC or telephony, while routing the assistant's tool calls through EVA's deterministic scenario database.

## Architecture

```
ElevenLabs User Sim → WebSocket → TelephonyBridgeServer → [Transport] → External Voice Assistant
                                                                              ↓ tool call webhook
                                              ngrok → ToolWebhookService → EVA's ToolExecutor (scenario DB)
```

## Transports

### WebRTC (`transport=webrtc`)

Connects to a **Telnyx AI Assistant** via the `@telnyx/webrtc` SDK. This is the same path a website widget user takes — lowest latency for Telnyx assistants.

**Requirements:**
- Telnyx assistant ID (existing or auto-created)
- Node.js (for the WebRTC helper subprocess)

### Call Control (`transport=call_control`)

Dials **any SIP URI or phone number** via Telnyx Call Control API and streams audio via media streaming WebSocket. Can benchmark:

- Telnyx AI Assistants (via SIP URI)
- Competitor voice agents (via phone number)
- Live IVR systems
- Any voice agent reachable by phone

**Requirements:**
- Telnyx API key
- Telnyx SIP Connection ID (for placing outbound calls)
- A public WebSocket URL for media streaming (e.g., ngrok)
- A caller ID / from number

## Setup

### 1. Install EVA

```bash
git clone https://github.com/team-telnyx/eva.git
cd eva
pip install -e .
```

### 2. Set up ngrok

The tool webhook service needs a public URL. Get a free static domain from [ngrok](https://ngrok.com):

```bash
# Start ngrok (tool webhooks)
ngrok http 8888 --url your-domain.ngrok-free.app
```

For Call Control transport, you also need a WebSocket tunnel for media streaming:

```bash
# Second tunnel for media streaming (separate terminal)
ngrok http 8889 --url your-stream-domain.ngrok-free.app
```

### 3. Configure environment

#### Option A: WebRTC with existing assistant

```bash
export EVA_MODEL__TRANSPORT="webrtc"
export EVA_MODEL__TELNYX_ASSISTANT_ID="your-assistant-uuid"
export EVA_MODEL__TELNYX_API_KEY="your-telnyx-api-key"
export EVA_MODEL__WEBHOOK_BASE_URL="https://your-domain.ngrok-free.app"
export EVA_MODEL__WEBHOOK_PORT=8888
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

#### Option B: WebRTC with auto-created assistant

EVA creates a Telnyx assistant from its agent config, runs the benchmark, and cleans up.

```bash
export EVA_MODEL__TRANSPORT="webrtc"
export EVA_MODEL__TELNYX_API_KEY="your-telnyx-api-key"
export EVA_MODEL__TELNYX_MODEL="moonshotai/Kimi-K2.5"
export EVA_MODEL__TELNYX_VOICE="Telnyx.Ultra.a7a59115-2425-4192-844c-1e98ec7d6877"
export EVA_MODEL__WEBHOOK_BASE_URL="https://your-domain.ngrok-free.app"
export EVA_MODEL__WEBHOOK_PORT=8888
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

#### Option C: Call Control (any phone number or SIP URI)

```bash
export EVA_MODEL__TRANSPORT="call_control"
export EVA_MODEL__SIP_URI="sip:assistant-uuid@sip.telnyx.com"
export EVA_MODEL__TELNYX_API_KEY="your-telnyx-api-key"
export EVA_MODEL__CALL_CONTROL_CONNECTION_ID="your-sip-connection-id"
export EVA_MODEL__CALL_CONTROL_FROM="+18005551234"
export EVA_MODEL__CALL_CONTROL_STREAM_URL="wss://your-stream-domain.ngrok-free.app"
export EVA_MODEL__WEBHOOK_BASE_URL="https://your-domain.ngrok-free.app"
export EVA_MODEL__WEBHOOK_PORT=8888
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

### 4. Configure assistant tools

For the tool webhook to work, the Telnyx assistant's tools must point at the webhook URL with `{{call_control_id}}` for routing:

```
https://your-domain.ngrok-free.app/tools/{{call_control_id}}/get_reservation
```

If using auto-creation (Option B), this is handled automatically.

### 5. Run the benchmark

```bash
python -m eva --config configs/your_config.yaml
```

## Tool Webhook

The `ToolWebhookService` runs alongside EVA and exposes the scenario database as HTTP endpoints:

```
POST /tools/{call_id}/{tool_name}
```

Each conversation registers its own `ToolExecutor` instance, keyed by `call_control_id`. The assistant's webhook tools hit the ngrok URL → route to the right scenario DB → return deterministic results.

This enables full EVA task completion scoring even for black-box external assistants.

## Audio Format

- EVA user simulator: μ-law 8kHz (Twilio-style WebSocket framing)
- Telnyx Call Control media streaming: μ-law 8kHz (PCMU)
- WebRTC: Opus (converted by the SDK)

The Call Control transport passes μ-law audio through with zero codec conversion.

## Metrics

All EVA metrics work with the telephony bridge:

- **Experience metrics** (audio/transcript-based): conciseness, turn-taking, latency, conversation progression
- **Task completion** (tool webhook): deterministic, uses EVA's scenario database
- **LLM judge metrics**: evaluated on transcripts

## Platform Defaults

When auto-creating assistants, EVA uses Telnyx platform defaults:

| Setting | Default |
|---------|---------|
| LLM | moonshotai/Kimi-K2.5 |
| STT | deepgram/flux |
| TTS Voice | Telnyx.Ultra.a7a59115-2425-4192-844c-1e98ec7d6877 |
| Interruption | Enabled (0.1s endpointing) |
| Voice speed | 1.0 |
| Expressive mode | Enabled |
| Language boost | English |
