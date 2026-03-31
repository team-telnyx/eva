"""Helpers for writing provider-normalized metrics artifacts."""

import json
from pathlib import Path
from typing import Any

from eva.utils.log_processing import group_consecutive_turns

AUDIT_ASSISTANT_CONTENT_KEY = "content"
AUDIT_ASSISTANT_ALREADY_SPOKEN_KEY = "already_spoken"


def build_metrics_audit_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize audit transcript entries for generic metrics processing."""
    normalized: list[dict[str, Any]] = []
    for entry in transcript:
        copied = entry.copy()
        if copied.get("message_type") == "assistant":
            text = _extract_text(copied.get("value"))
            if text:
                copied["value"] = {
                    AUDIT_ASSISTANT_CONTENT_KEY: text,
                    AUDIT_ASSISTANT_ALREADY_SPOKEN_KEY: True,
                }
        normalized.append(copied)
    return normalized


def build_intended_assistant_messages(
    transcript: list[dict[str, Any]],
    intended_speech: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge provider-native intended speech with audit-log fallbacks."""
    fallback_messages = [
        {
            "text": text,
            "timestamp_ms": timestamp_ms,
        }
        for entry in transcript
        if entry.get("message_type") == "assistant"
        if (text := _extract_text(entry.get("value")))
        if (timestamp_ms := _extract_timestamp_ms(entry.get("timestamp"))) is not None
    ]

    merged: list[dict[str, Any]] = []
    total = max(len(intended_speech), len(fallback_messages))
    for index in range(total):
        provider_message = intended_speech[index] if index < len(intended_speech) else None
        fallback_message = fallback_messages[index] if index < len(fallback_messages) else None
        message = provider_message or fallback_message
        if message is None:
            continue
        text = str(message.get("text", "")).strip()
        timestamp_ms = _extract_timestamp_ms(message.get("timestamp_ms"))
        if not text or timestamp_ms is None:
            continue
        merged.append({"text": text, "timestamp_ms": timestamp_ms})
    return merged


def build_message_trace(
    transcript: list[dict[str, Any]],
    intended_assistant_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a message-level trace for judge metrics."""
    trace: list[dict[str, Any]] = []

    for entry in transcript:
        timestamp_ms = _extract_timestamp_ms(entry.get("timestamp"))
        if timestamp_ms is None:
            continue

        message_type = entry.get("message_type")
        value = entry.get("value")
        if message_type == "user":
            text = _extract_text(value)
            if text:
                trace.append(
                    {
                        "role": "user",
                        "content": text,
                        "timestamp": timestamp_ms,
                        "type": "transcribed",
                    }
                )
        elif message_type == "tool_call" and isinstance(value, dict):
            trace.append(
                {
                    "tool_name": value.get("tool"),
                    "parameters": value.get("parameters"),
                    "timestamp": timestamp_ms,
                    "type": "tool_call",
                }
            )
        elif message_type == "tool_response" and isinstance(value, dict):
            trace.append(
                {
                    "tool_name": value.get("tool"),
                    "tool_response": value.get("response"),
                    "timestamp": timestamp_ms,
                    "type": "tool_response",
                }
            )

    for message in intended_assistant_messages:
        timestamp_ms = _extract_timestamp_ms(message.get("timestamp_ms"))
        text = str(message.get("text", "")).strip()
        if timestamp_ms is None or not text:
            continue
        trace.append(
            {
                "role": "assistant",
                "content": text,
                "timestamp": timestamp_ms,
                "type": "intended",
            }
        )

    trace.sort(key=lambda entry: (entry["timestamp"], _trace_priority(entry)))
    _assign_turn_ids(trace)
    return group_consecutive_turns(trace)


def write_message_trace(path: Path, trace: list[dict[str, Any]]) -> None:
    """Persist message trace entries as JSONL."""
    with open(path, "w", encoding="utf-8") as file_obj:
        for entry in trace:
            file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _assign_turn_ids(trace: list[dict[str, Any]]) -> None:
    current_turn = 0
    assistant_spoke_since_user = False

    for entry in trace:
        role = entry.get("role")
        if role == "user":
            if assistant_spoke_since_user:
                current_turn += 1
                assistant_spoke_since_user = False
            entry["turn_id"] = current_turn
            continue

        entry["turn_id"] = current_turn
        if role == "assistant":
            assistant_spoke_since_user = True


def _extract_text(value: Any) -> str:
    if isinstance(value, dict):
        content = value.get(AUDIT_ASSISTANT_CONTENT_KEY)
        if isinstance(content, str):
            return content.strip()
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_priority(entry: dict[str, Any]) -> int:
    if entry.get("role") == "assistant":
        return 0
    if entry.get("role") == "user":
        return 1
    if entry.get("type") == "tool_call":
        return 2
    if entry.get("type") == "tool_response":
        return 3
    return 5
