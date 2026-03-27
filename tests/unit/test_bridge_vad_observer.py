"""Unit tests for the bridge VAD observer."""

import json
from pathlib import Path

import pytest
from pipecat.audio.vad.vad_analyzer import VADState

from eva.assistant.agentic.audit_log import AuditLog
from eva.assistant.bridge_vad_observer import (
    TURN_MERGE_GAP_SECONDS,
    BridgeVADObserver,
)


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
async def test_vad_transitions_with_coalescing_emit_one_turn(tmp_path: Path):
    """Speech→quiet→speech within merge gap should produce a single audio_start/end pair."""
    output_dir = tmp_path / "output"
    clock = _FakeClock(1000.0)

    # User VAD: speak, quiet briefly (within merge gap), speak again, then long silence
    user_states = [
        VADState.STARTING,   # 1001.0 — open turn
        VADState.SPEAKING,   # 1001.1
        VADState.STOPPING,   # 1001.2 — silence starts
        VADState.QUIET,      # 1001.3 — silence (0.1s so far, < merge gap)
        VADState.QUIET,      # 1001.4 — silence (0.2s, still < merge gap)
        VADState.STARTING,   # 1001.5 — speech resumes, same turn!
        VADState.SPEAKING,   # 1001.6
        VADState.STOPPING,   # 1001.7 — silence starts again
        VADState.QUIET,      # 1001.8
    ]
    # Feed enough QUIET states to exceed the merge gap
    gap_chunks = int(TURN_MERGE_GAP_SECONDS / 0.1) + 2
    for _ in range(gap_chunks):
        user_states.append(VADState.QUIET)

    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["hello world"]),
        AuditLog(),
        vad_factory=_make_vad_factory(user_states, []),
        time_provider=clock,
    )

    base = 1001.0
    for i in range(len(user_states)):
        ts = base + i * 0.1
        await observer.feed_user_audio(b"\x01\x02" * 10, ts)

    clock.now = 1010.0
    await observer.finalize()

    events = _load_events(output_dir)
    audio_events = [e for e in events if "event_type" in e]

    # Should be exactly ONE audio_start and ONE audio_end (coalesced turn)
    starts = [e for e in audio_events if e["event_type"] == "audio_start"]
    ends = [e for e in audio_events if e["event_type"] == "audio_end"]
    assert len(starts) == 1, f"Expected 1 audio_start, got {len(starts)}"
    assert len(ends) == 1, f"Expected 1 audio_end, got {len(ends)}"
    assert starts[0]["user"] == "elevenlabs_user"
    assert starts[0]["audio_timestamp"] == 1001.0  # Turn opened at first STARTING


@pytest.mark.asyncio
async def test_long_silence_separates_turns(tmp_path: Path):
    """Two speech bursts separated by > TURN_MERGE_GAP_SECONDS should be two turns."""
    output_dir = tmp_path / "output"
    clock = _FakeClock(2000.0)

    gap_chunks = int(TURN_MERGE_GAP_SECONDS / 0.1) + 2
    user_states = (
        [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks  # Long silence → closes first turn
        + [VADState.STARTING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks  # Long silence → closes second turn
    )

    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["first turn", "second turn"]),
        AuditLog(),
        vad_factory=_make_vad_factory(user_states, []),
        time_provider=clock,
    )

    base = 2001.0
    for i in range(len(user_states)):
        await observer.feed_user_audio(b"\x01\x02" * 10, base + i * 0.1)

    clock.now = 2100.0
    await observer.finalize()

    events = _load_events(output_dir)
    starts = [e for e in events if e.get("event_type") == "audio_start" and e["user"] == "elevenlabs_user"]
    ends = [e for e in events if e.get("event_type") == "audio_end" and e["user"] == "elevenlabs_user"]
    assert len(starts) == 2, f"Expected 2 turns, got {len(starts)}"
    assert len(ends) == 2


@pytest.mark.asyncio
async def test_user_and_assistant_transcription_and_audit_log(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(3000.0)
    audit_log = AuditLog()

    gap_chunks = int(TURN_MERGE_GAP_SECONDS / 0.1) + 2

    # Use 0.2s spacing so speech spans are long enough (>= MIN_SEGMENT_DURATION_SECONDS)
    user_states = (
        [VADState.STARTING, VADState.SPEAKING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks
    )
    asst_states = (
        [VADState.STARTING, VADState.SPEAKING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks
    )

    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["hello from user", "hello from assistant"]),
        audit_log,
        vad_factory=_make_vad_factory(user_states, asst_states),
        time_provider=clock,
    )

    base_u = 3001.0
    for i in range(len(user_states)):
        await observer.feed_user_audio(b"\x01\x02" * 10, base_u + i * 0.2)

    base_a = 3010.0
    for i in range(len(asst_states)):
        await observer.feed_assistant_audio(b"\x03\x04" * 10, base_a + i * 0.2)

    clock.now = 3100.0
    await observer.finalize()

    events = _load_events(output_dir)
    # Monotonic sequences
    seqs = [e["sequence"] for e in events]
    assert seqs == list(range(1, len(events) + 1))

    # Speech events present
    speech_events = [e for e in events if e.get("type") in {"user_speech", "assistant_speech"}]
    assert len(speech_events) == 2
    assert speech_events[0]["type"] == "user_speech"
    assert speech_events[0]["data"]["text"] == "hello from user"
    assert speech_events[1]["type"] == "assistant_speech"
    assert speech_events[1]["data"]["text"] == "hello from assistant"

    # Audit log entries
    assert sorted(entry["message_type"] for entry in audit_log.transcript) == ["assistant", "user"]

    # Session ended last
    assert events[-1]["data"]["state"] == "session_ended"


@pytest.mark.asyncio
async def test_response_latency_from_coalesced_turns(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(4000.0)

    gap_chunks = int(TURN_MERGE_GAP_SECONDS / 0.1) + 2

    user_states = (
        [VADState.STARTING, VADState.SPEAKING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks
    )
    asst_states = (
        [VADState.STARTING, VADState.SPEAKING, VADState.SPEAKING, VADState.STOPPING]
        + [VADState.QUIET] * gap_chunks
    )

    observer = BridgeVADObserver(
        output_dir,
        _FakeTranscriber(["", ""]),
        AuditLog(),
        vad_factory=_make_vad_factory(user_states, asst_states),
        time_provider=clock,
    )

    # User speaks 4001.0 - 4001.6 (4 states at 0.2s spacing), silence starts at 4001.6
    base_u = 4001.0
    for i in range(len(user_states)):
        await observer.feed_user_audio(b"\x01\x02" * 10, base_u + i * 0.2)

    # User turn closes after merge gap. Silence started at 4001.6 (STOPPING).
    # Assistant speaks starting at 4010.0
    base_a = 4010.0
    for i in range(len(asst_states)):
        await observer.feed_assistant_audio(b"\x03\x04" * 10, base_a + i * 0.2)

    clock.now = 4100.0
    await observer.finalize()

    latencies = json.loads((output_dir / "response_latencies.json").read_text())
    assert latencies["count"] == 1
    assert latencies["source"] == "vad_observer"
    # Latency = assistant start (4010.0) - user turn end (silence started at 4001.6)
    assert latencies["latencies"][0] == pytest.approx(8.4, abs=0.01)


@pytest.mark.asyncio
async def test_finalize_closes_open_turn_and_transcribes(tmp_path: Path):
    output_dir = tmp_path / "output"
    clock = _FakeClock(5000.0)
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

    await observer.feed_user_audio(b"\x01\x02" * 10, 5001.0)
    await observer.feed_user_audio(b"\x01\x02" * 10, 5001.1)

    clock.now = 5010.0
    await observer.finalize()

    events = _load_events(output_dir)
    # Last events: audio_end, user_speech, session_ended
    assert events[-1]["data"]["state"] == "session_ended"
    assert events[-2]["type"] == "user_speech"
    assert events[-2]["data"]["text"] == "trailing user"
    assert events[-3]["event_type"] == "audio_end"
