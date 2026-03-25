#!/usr/bin/env python3
"""Quick test: place a Call Control call, capture 10s of audio, save as WAV."""
import asyncio
import base64
import json
import struct
import wave

import aiohttp


async def main():
    for line in open("/Users/jamesw/.telnyx-credentials"):
        line = line.strip()
        if line.startswith("TELNYX_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break
    else:
        raise ValueError("TELNYX_API_KEY not found")
    
    captured = bytearray()
    connected = asyncio.Event()
    
    # Start a tiny WS server to receive media stream
    from fastapi import FastAPI, WebSocket
    import uvicorn
    
    app = FastAPI()
    
    @app.websocket("/media-stream/test")
    async def media_stream(websocket: WebSocket):
        await websocket.accept()
        print("Media stream connected!")
        connected.set()
        count = 0
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                evt = msg.get("event")
                if evt == "media":
                    payload = msg.get("media", {}).get("payload", "")
                    if payload:
                        audio = base64.b64decode(payload)
                        captured.extend(audio)
                        count += 1
                        if count <= 3:
                            print(f"  chunk #{count}: {len(audio)} bytes, first 4: {audio[:4].hex()}")
                elif evt == "start":
                    print(f"Stream started: {msg.get('stream_id')}")
                elif evt == "connected":
                    print("Stream signaled connected")
                elif evt == "stop":
                    print("Stream stopped")
                    break
        except Exception as e:
            print(f"Stream error: {e}")
    
    # Start server
    config = uvicorn.Config(app, host="0.0.0.0", port=8888, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    
    # Place call
    async with aiohttp.ClientSession() as session:
        print("Placing call...")
        async with session.post(
            "https://api.telnyx.com/v2/calls",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "connection_id": "2923665753846581126",
                "to": "+13125726122",
                "from": "+19293799793",
                "stream_url": "wss://umbiliform-kasandra-unlet.ngrok-free.dev/media-stream/test",
                "stream_track": "both_tracks",
                "stream_bidirectional_mode": "rtp",
                "stream_bidirectional_codec": "PCMU",
            },
        ) as resp:
            body = await resp.json()
            ccid = body.get("data", {}).get("call_control_id")
            print(f"Call placed: {ccid}, status={resp.status}")
            if resp.status >= 400:
                print(f"Error: {json.dumps(body)}")
                server.should_exit = True
                return
        
        # Wait for media stream
        await asyncio.wait_for(connected.wait(), timeout=30)
        print(f"Capturing 10 seconds of audio...")
        await asyncio.sleep(10)
        
        # Hangup
        print(f"Hanging up...")
        async with session.post(
            f"https://api.telnyx.com/v2/calls/{ccid}/actions/hangup",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={},
        ) as resp:
            print(f"Hangup: {resp.status}")
    
    await asyncio.sleep(1)
    server.should_exit = True
    
    # Save as WAV (μ-law 8kHz)
    print(f"\nCaptured {len(captured)} bytes of μ-law audio")
    
    # Convert μ-law to PCM for WAV
    try:
        import audioop
    except ImportError:
        import audioop_lts as audioop
    
    pcm = audioop.ulaw2lin(bytes(captured), 2)
    
    with wave.open("/tmp/eva_test_capture.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    
    print(f"Saved to /tmp/eva_test_capture.wav ({len(pcm)} bytes PCM, {len(pcm)/16000:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
