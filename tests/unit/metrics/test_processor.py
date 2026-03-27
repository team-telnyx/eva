"""Focused tests for generic processor fallback behavior."""

from eva.metrics.processor import MetricsContextProcessor, _ProcessorContext, _validate_conversation_trace


def test_validate_conversation_trace_preserves_audit_assistant_without_pipecat_text():
    ctx = _ProcessorContext()
    ctx.record_id = "bridge-no-pipecat"

    trace = [
        {
            "role": "assistant",
            "content": "Sure, let me help with that.",
            "type": "intended",
            "turn_id": 1,
            "_audit_source": True,
        }
    ]

    assert _validate_conversation_trace(trace, ctx) == [
        {
            "role": "assistant",
            "content": "Sure, let me help with that.",
            "type": "intended",
            "turn_id": 1,
        }
    ]


def test_validate_conversation_trace_still_truncates_when_pipecat_text_exists():
    ctx = _ProcessorContext()
    ctx.record_id = "pipecat-truncation"
    ctx._intended_assistant_segments = {
        1: ["Sure, I can help you with that."]
    }

    trace = [
        {
            "role": "assistant",
            "content": "Sure, I can help you with that. Could you please provide your confirmation number?",
            "type": "intended",
            "turn_id": 1,
            "_audit_source": True,
        }
    ]

    assert _validate_conversation_trace(trace, ctx) == [
        {
            "role": "assistant",
            "content": "Sure, I can help you with that.",
            "type": "intended",
            "turn_id": 1,
        }
    ]


def test_extract_turns_uses_audit_log_assistant_as_generic_fallback():
    ctx = _ProcessorContext()
    ctx.record_id = "bridge-fallback"
    ctx.history = [
        {"timestamp_ms": 1000, "source": "audit_log", "event_type": "assistant", "data": "Welcome to EVA."},
        {
            "timestamp_ms": 1000,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.0},
        },
        {
            "timestamp_ms": 1400,
            "source": "elevenlabs",
            "event_type": "assistant_speech",
            "data": {"data": {"text": "Welcome to EVA."}},
        },
        {
            "timestamp_ms": 1600,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 1.6},
        },
        {
            "timestamp_ms": 2000,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 2.0},
        },
        {"timestamp_ms": 2050, "source": "audit_log", "event_type": "user", "data": "I need help"},
        {
            "timestamp_ms": 2100,
            "source": "elevenlabs",
            "event_type": "user_speech",
            "data": {"data": {"text": "I need help"}},
        },
        {
            "timestamp_ms": 2400,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "elevenlabs_user", "audio_timestamp": 2.4},
        },
        {"timestamp_ms": 3000, "source": "audit_log", "event_type": "assistant", "data": "I found your reservation."},
        {
            "timestamp_ms": 3000,
            "source": "elevenlabs",
            "event_type": "audio_start",
            "data": {"user": "pipecat_agent", "audio_timestamp": 3.0},
        },
        {
            "timestamp_ms": 3400,
            "source": "elevenlabs",
            "event_type": "assistant_speech",
            "data": {"data": {"text": "I found your reservation."}},
        },
        {
            "timestamp_ms": 3600,
            "source": "elevenlabs",
            "event_type": "audio_end",
            "data": {"user": "pipecat_agent", "audio_timestamp": 3.6},
        },
        {
            "timestamp_ms": 4000,
            "source": "elevenlabs",
            "event_type": "connection_state",
            "data": {"data": {"state": "session_ended"}},
        },
    ]

    MetricsContextProcessor._extract_turns_from_history(ctx)
    MetricsContextProcessor._reconcile_transcript_with_tools(ctx)

    assert ctx.intended_assistant_turns == {
        0: "Welcome to EVA.",
        1: "I found your reservation. [speaker likely cut itself off]",
    }
    assert ctx.transcribed_assistant_turns == {
        0: "Welcome to EVA.",
        1: "I found your reservation. [speaker likely cut itself off]",
    }
    assert ctx.transcribed_user_turns == {
        1: "I need help",
    }
    assert ctx.conversation_trace == [
        {"role": "assistant", "content": "Welcome to EVA.", "timestamp": 1000, "type": "intended", "turn_id": 0},
        {"role": "user", "content": "I need help", "timestamp": 2050, "type": "transcribed", "turn_id": 1},
        {
            "role": "assistant",
            "content": "I found your reservation. [speaker likely cut itself off]",
            "timestamp": 3000,
            "type": "intended",
            "turn_id": 1,
        },
    ]
