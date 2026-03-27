"""Unit tests for the bridge VAD observer."""

import json
from pathlib import Path

import pytest
from pipecat.audio.vad.vad_analyzer import VADState

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.bridge_vad_observer import BridgeVADObserver


class _FakeClock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


class _FakeTranscriber:
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self.calls: list[tuple[bytes, int]] = []

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        self.calls.append((audio_data, sample_rate))
        return next(self._responses, "")


class _FakeVAD:
    def __init__(self, states: list[VADState]):
        self._states = iter(states)
        self.sample_rate: int | None = None

    def set_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    async def analyze_audio(self, _pcm_bytes: bytes) -> VADState:
        return next(self._states)


def _make_vad_factory(*stream_states: list[VADState]):
    states = list(stream_states)

    def factory():
        return _FakeVAD(states.pop(0))

    return factory


def _load_events(output_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (output_dir / "elevenlabs_events.jsonl").read_text().splitlines()]


@pytest.mark.asyncio
async def test_vad_state_transitions_emit_exact_audio_event_schema(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(1000.0)
    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber([""]),
        AuditLog(),
        vad_factory=_make_vad_factory(
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
            [],
        ),
        time_provider=clock,
    )

    await observer.feed_user_audio(b"\x01\x02" * 10, 1001.0)
    await observer.feed_user_audio(b"\x01\x02" * 10, 1001.1)
    await observer.feed_user_audio(b"\x01\x02" * 10, 1001.2)
    await observer.feed_user_audio(b"\x01\x02" * 10, 1001.3)
    clock.now = 1002.0
    await observer.finalize()

    assert _load_events(output_dir) == [
        {"timestamp": 1000000, "sequence": 1, "type": "connection_state", "data": {"state": "connected", "details": {}}},
        {
            "timestamp": 1000000,
            "sequence": 2,
            "type": "connection_state",
            "data": {"state": "session_started", "details": {}},
        },
        {
            "timestamp": 1001100,
            "sequence": 3,
            "event_type": "audio_start",
            "user": "elevenlabs_user",
            "audio_timestamp": 1001.1,
        },
        {
            "timestamp": 1001300,
            "sequence": 4,
            "event_type": "audio_end",
            "user": "elevenlabs_user",
            "audio_timestamp": 1001.3,
        },
        {
            "timestamp": 1002000,
            "sequence": 5,
            "type": "connection_state",
            "data": {"state": "session_ended", "details": {}},
        },
    ]


@pytest.mark.asyncio
async def test_user_and_assistant_segments_are_transcribed_and_sequence_is_monotonic(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(2000.0)
    audit_log = AuditLog()
    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["hello from user", "hello from assistant"]),
        audit_log,
        vad_factory=_make_vad_factory(
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
        ),
        time_provider=clock,
    )

    await observer.feed_user_audio(b"\x01\x02" * 10, 2001.0)
    await observer.feed_user_audio(b"\x01\x02" * 10, 2001.1)
    await observer.feed_user_audio(b"\x01\x02" * 10, 2001.2)
    await observer.feed_user_audio(b"\x01\x02" * 10, 2001.3)

    await observer.feed_assistant_audio(b"\x03\x04" * 10, 2002.0)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 2002.1)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 2002.2)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 2002.3)

    clock.now = 2003.0
    await observer.finalize()

    events = _load_events(output_dir)
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    speech_events = [event for event in events if event.get("type") in {"user_speech", "assistant_speech"}]
    assert speech_events == [
        {
            "timestamp": 2003000,
            "sequence": speech_events[0]["sequence"],
            "type": "user_speech",
            "data": {"text": "hello from user", "source": "elevenlabs_agent"},
        },
        {
            "timestamp": 2003000,
            "sequence": speech_events[1]["sequence"],
            "type": "assistant_speech",
            "data": {"text": "hello from assistant", "source": "pipecat_assistant"},
        },
    ]
    assert events[-1]["data"]["state"] == "session_ended"
    assert sorted(entry["message_type"] for entry in audit_log.transcript) == ["assistant", "user"]
    assert observer.get_transcript_entries() == [
        {"timestamp": "1970-01-01T00:33:21.100000+00:00", "role": "user", "content": "hello from user"},
        {
            "timestamp": "1970-01-01T00:33:22.100000+00:00",
            "role": "assistant",
            "content": "hello from assistant",
        },
    ]


@pytest.mark.asyncio
async def test_response_latencies_written_from_vad_boundaries(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(3000.0)
    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["", ""]),
        AuditLog(),
        vad_factory=_make_vad_factory(
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
            [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING, VADState.QUIET],
        ),
        time_provider=clock,
    )

    await observer.feed_user_audio(b"\x01\x02" * 10, 3001.0)
    await observer.feed_user_audio(b"\x01\x02" * 10, 3001.1)
    await observer.feed_user_audio(b"\x01\x02" * 10, 3001.2)
    await observer.feed_user_audio(b"\x01\x02" * 10, 3001.3)

    await observer.feed_assistant_audio(b"\x03\x04" * 10, 3002.0)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 3002.1)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 3002.2)
    await observer.feed_assistant_audio(b"\x03\x04" * 10, 3002.3)

    clock.now = 3003.0
    await observer.finalize()

    latencies = json.loads((output_dir / "response_latencies.json").read_text())
    assert latencies == {
        "latencies": [0.8],
        "mean": 0.8,
        "max": 0.8,
        "min": 0.8,
        "count": 1,
        "source": "vad_observer",
    }


@pytest.mark.asyncio
async def test_finalize_closes_open_segment_and_writes_session_ended(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(4000.0)
    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["trailing user"]),
        AuditLog(),
        vad_factory=_make_vad_factory(
            [VADState.STARTING, VADState.SPEAKING],
            [],
        ),
        time_provider=clock,
    )

    await observer.feed_user_audio(b"\x01\x02" * 10, 4001.0)
    await observer.feed_user_audio(b"\x01\x02" * 10, 4001.1)

    clock.now = 4002.0
    await observer.finalize()

    events = _load_events(output_dir)
    assert events[-1] == {
        "timestamp": 4002000,
        "sequence": 6,
        "type": "connection_state",
        "data": {"state": "session_ended", "details": {}},
    }
    assert events[-2] == {
        "timestamp": 4002000,
        "sequence": 5,
        "type": "user_speech",
        "data": {"text": "trailing user", "source": "elevenlabs_agent"},
    }
