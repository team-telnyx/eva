#!/usr/bin/env python3
"""Rebuild audit_log.json and pipecat_logs.jsonl from the Telnyx Conversations API.

Fetches full message history for each conversation and writes them in the format
EVA's metrics processor expects.
"""

import json
import os
import sys
from pathlib import Path

import requests

API_KEY = os.environ["TELNYX_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE_URL = "https://api.telnyx.com/v2/ai/conversations"

# Map conversation IDs to record IDs
CONV_MAP = {
    "8d1428a9-421d-496b-8707-7657445a4113": "1.1.2",
    "6b5e142f-4925-4d79-b99a-97950606b568": "1.1.3",
    "360c9749-1b98-4079-aa93-0352b464a2ca": "1.1.4",
}

RUN_DIR = Path("output/2026-03-26_01-30-30.610790")


def fetch_messages(conv_id: str) -> list[dict]:
    """Fetch all messages for a conversation, handling pagination."""
    all_msgs = []
    url = f"{BASE_URL}/{conv_id}/messages?page%5Bsize%5D=100"
    while url:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        msgs = [m for m in data.get("data", []) if m is not None]
        all_msgs.extend(msgs)
        # Check for next page
        meta = data.get("meta", {})
        url = meta.get("next_page_url")
    # API returns reverse chronological; flip to chronological
    all_msgs.reverse()
    return all_msgs


def messages_to_audit_transcript(messages: list[dict]) -> list[dict]:
    """Convert API messages to audit_log transcript format."""
    transcript = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        text = msg.get("text", "")
        tool_calls = msg.get("tool_calls") or []
        tool_call_id = msg.get("tool_call_id")
        meta = msg.get("metadata") or {}
        sent_at = msg.get("sent_at", msg.get("created_at", ""))

        # Convert sent_at to epoch ms
        ts = 0
        if sent_at:
            from datetime import datetime, timezone

            try:
                dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                ts = int(dt.timestamp() * 1000)
            except (ValueError, TypeError):
                ts = i  # fallback ordering

        if role == "user" and text:
            transcript.append(
                {
                    "timestamp": ts,
                    "message_type": "user",
                    "type": "user",
                    "displayName": "User",
                    "value": text,
                }
            )
        elif role == "assistant":
            if text:
                entry = {
                    "timestamp": ts,
                    "message_type": "assistant",
                    "type": "assistant",
                    "displayName": "Assistant",
                    "value": text,
                }
                # Add latency metadata if available
                if meta.get("end_user_perceived_latency_ms"):
                    entry["latency"] = {
                        "end_user_perceived_ms": meta["end_user_perceived_latency_ms"],
                        "transcription_ms": meta.get("transcription_duration_ms", 0),
                        "llm_first_token_ms": meta.get("llm_first_token_duration_ms", 0),
                        "audio_first_token_ms": meta.get("audio_first_token_duration_ms", 0),
                    }
                transcript.append(entry)
            for tc in tool_calls:
                fn = tc.get("function", {})
                # Parse arguments string to dict
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                transcript.append(
                    {
                        "timestamp": ts,
                        "message_type": "tool_call",
                        "type": "tool_call",
                        "displayName": "Tool",
                        "value": {
                            "tool": fn.get("name", ""),
                            "parameters": args,
                            "tool_call_id": tc.get("id", ""),
                        },
                    }
                )
        elif role == "tool":
            # Find the matching tool_call to get the tool name
            tool_name = ""
            if tool_call_id:
                for prev in reversed(transcript):
                    if (
                        prev.get("message_type") == "tool_call"
                        and prev.get("value", {}).get("tool_call_id") == tool_call_id
                    ):
                        tool_name = prev["value"]["tool"]
                        break
            transcript.append(
                {
                    "timestamp": ts,
                    "message_type": "tool_response",
                    "type": "tool_response",
                    "displayName": "Tool Response",
                    "value": {
                        "tool": tool_name,
                        "response": text,
                        "tool_call_id": tool_call_id or "",
                    },
                }
            )

    return transcript


def messages_to_pipecat_logs(messages: list[dict]) -> list[dict]:
    """Generate pipecat_logs.jsonl entries from assistant speech messages."""
    logs = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = msg.get("text", "")
        if not text:
            continue
        sent_at = msg.get("sent_at", msg.get("created_at", ""))
        ts = 0
        if sent_at:
            from datetime import datetime

            try:
                dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
                ts = int(dt.timestamp() * 1000)
            except (ValueError, TypeError):
                pass
        logs.append({"type": "tts_text", "start_timestamp": ts, "data": {"frame": text}})
    return logs


def messages_to_response_latencies(messages: list[dict]) -> dict:
    """Extract response latencies from assistant message metadata."""
    latencies = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        meta = msg.get("metadata") or {}
        latency_ms = meta.get("end_user_perceived_latency_ms")
        if latency_ms and latency_ms > 0:
            latencies.append(latency_ms / 1000.0)

    if not latencies:
        return {"latencies": [], "mean": 0, "max": 0, "min": 0, "count": 0}

    return {
        "latencies": latencies,
        "mean": sum(latencies) / len(latencies),
        "max": max(latencies),
        "min": min(latencies),
        "count": len(latencies),
    }


def main():
    for conv_id, record_id in CONV_MAP.items():
        record_dir = RUN_DIR / "records" / record_id
        if not record_dir.exists():
            print(f"⚠️  {record_id}: directory not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {record_id} (conversation {conv_id})")

        messages = fetch_messages(conv_id)
        print(f"  Fetched {len(messages)} messages from API")

        # Rebuild audit_log.json — merge API transcript with existing tool events
        transcript = messages_to_audit_transcript(messages)
        audit_path = record_dir / "audit_log.json"
        existing_audit = {}
        if audit_path.exists():
            existing_audit = json.load(open(audit_path))

        # Keep existing metadata, replace transcript
        existing_audit["transcript"] = transcript
        existing_audit["source"] = "telnyx_conversations_api"
        with open(audit_path, "w") as f:
            json.dump(existing_audit, f, indent=2)
        print(f"  Wrote audit_log.json ({len(transcript)} events)")

        # Generate pipecat_logs.jsonl
        pipecat_logs = messages_to_pipecat_logs(messages)
        pipecat_path = record_dir / "pipecat_logs.jsonl"
        with open(pipecat_path, "w") as f:
            for log in pipecat_logs:
                f.write(json.dumps(log) + "\n")
        print(f"  Wrote pipecat_logs.jsonl ({len(pipecat_logs)} entries)")

        # Generate response_latencies.json
        latencies = messages_to_response_latencies(messages)
        latencies_path = record_dir / "response_latencies.json"
        with open(latencies_path, "w") as f:
            json.dump(latencies, f, indent=2)
        print(
            f"  Wrote response_latencies.json ({latencies['count']} measurements, "
            f"mean={latencies['mean']:.3f}s)"
        )

        # Summary
        roles = {}
        for t in transcript:
            mt = t["message_type"]
            roles[mt] = roles.get(mt, 0) + 1
        print(f"  Transcript breakdown: {roles}")


if __name__ == "__main__":
    main()
