#!/usr/bin/env node
/*
 * Telnyx's published WebRTC SDK exposes a Node entrypoint but still references
 * browser globals at runtime. This helper supplies the minimal shims needed to
 * run the SDK under Node and bridges raw PCM audio over a localhost WebSocket.
 *
 * The older `wrtc` package did not install cleanly on the current Node runtime
 * in this environment, so this helper uses the maintained `@roamhq/wrtc` fork.
 */

const process = require("node:process");
const { setInterval, clearInterval, setTimeout, clearTimeout } = require("node:timers");
const { WebSocketServer, WebSocket } = require("ws");
const wrtc = require("@roamhq/wrtc");
const { TelnyxRTC } = require("@telnyx/webrtc");

const PCM_SAMPLE_RATE = 8000;
const WEBRTC_SAMPLE_RATE = 48000;
const PCM_SAMPLE_WIDTH = 2;
const CHANNEL_COUNT = 1;
const WS_HOST = "127.0.0.1";
const AUDIO_FRAME_MS = 10;
const PCM_FRAME_SAMPLES = (PCM_SAMPLE_RATE * AUDIO_FRAME_MS) / 1000;
const WEBRTC_FRAME_SAMPLES = (WEBRTC_SAMPLE_RATE * AUDIO_FRAME_MS) / 1000;
const PCM_FRAME_BYTES = PCM_FRAME_SAMPLES * PCM_SAMPLE_WIDTH;
const UPSAMPLE_FACTOR = WEBRTC_SAMPLE_RATE / PCM_SAMPLE_RATE;
const DOWNSAMPLE_FACTOR = WEBRTC_SAMPLE_RATE / PCM_SAMPLE_RATE;

function log(level, message, extra = undefined) {
  const timestamp = new Date().toISOString();
  const suffix = extra === undefined ? "" : ` ${JSON.stringify(extra)}`;
  const line = `[${timestamp}] ${level.toUpperCase()} ${message}${suffix}`;
  if (level === "error") {
    console.error(line);
    return;
  }
  console.error(line);
}

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      args[key] = "true";
      continue;
    }
    args[key] = value;
    index += 1;
  }
  return args;
}

function ensureRequiredArg(args, key) {
  const value = args[key];
  if (!value) {
    throw new Error(`Missing required argument --${key}`);
  }
  return value;
}

function createLocalStorage() {
  const store = new Map();
  return {
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(key, String(value));
    },
    removeItem(key) {
      store.delete(key);
    },
  };
}

function installBrowserShims(localStream) {
  const localStorage = createLocalStorage();
  const document = {
    cookie: "",
    getElementById() {
      return null;
    },
    querySelector() {
      return null;
    },
    createElement() {
      return null;
    },
  };

  const navigator = {
    userAgent: "node",
    mediaDevices: {
      async getUserMedia() {
        return localStream;
      },
      async enumerateDevices() {
        return [];
      },
    },
  };

  const windowObject = {
    addEventListener() {},
    removeEventListener() {},
    clearInterval,
    clearTimeout,
    document,
    localStorage,
    navigator,
    setInterval,
    setTimeout,
  };

  globalThis.window = windowObject;
  globalThis.document = document;
  globalThis.localStorage = localStorage;
  globalThis.navigator = navigator;
  globalThis.WebSocket = WebSocket;
  globalThis.MediaStream = wrtc.MediaStream;
  globalThis.MediaStreamTrack = wrtc.MediaStreamTrack;
  globalThis.RTCPeerConnection = wrtc.RTCPeerConnection;
  globalThis.RTCSessionDescription = wrtc.RTCSessionDescription;
  globalThis.RTCIceCandidate = wrtc.RTCIceCandidate;
  globalThis.RTCRtpReceiver = wrtc.RTCRtpReceiver;
  globalThis.performance = {
    mark() {},
    measure() {},
    now() {
      return Date.now();
    },
  };
  windowObject.performance = globalThis.performance;
}

function pcmBufferToSamples(buffer) {
  const sampleCount = Math.floor(buffer.length / PCM_SAMPLE_WIDTH);
  return new Int16Array(buffer.buffer, buffer.byteOffset, sampleCount);
}

function cloneSamples(samples) {
  const copy = new Int16Array(samples.length);
  copy.set(samples);
  return copy;
}

function upsample8kTo48k(samples8k) {
  const samples48k = new Int16Array(samples8k.length * UPSAMPLE_FACTOR);
  let outputIndex = 0;
  for (let index = 0; index < samples8k.length; index += 1) {
    const value = samples8k[index];
    for (let repeat = 0; repeat < UPSAMPLE_FACTOR; repeat += 1) {
      samples48k[outputIndex] = value;
      outputIndex += 1;
    }
  }
  return samples48k;
}

function downsample48kTo8k(samples48k) {
  const sampleCount = Math.floor(samples48k.length / DOWNSAMPLE_FACTOR);
  const samples8k = new Int16Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    let total = 0;
    for (let offset = 0; offset < DOWNSAMPLE_FACTOR; offset += 1) {
      total += samples48k[index * DOWNSAMPLE_FACTOR + offset];
    }
    samples8k[index] = Math.max(-32768, Math.min(32767, Math.round(total / DOWNSAMPLE_FACTOR)));
  }
  return samples8k;
}

function samplesToBuffer(samples) {
  return Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength);
}

class AudioSourceBridge {
  constructor(audioSource) {
    this.audioSource = audioSource;
    this.pendingBytes = Buffer.alloc(0);
    this.timer = null;
  }

  enqueuePcm8k(buffer) {
    if (!buffer.length) {
      return;
    }
    this.pendingBytes = Buffer.concat([this.pendingBytes, buffer]);
  }

  start() {
    if (this.timer !== null) {
      return;
    }
    this.timer = setInterval(() => {
      let frameBuffer;
      if (this.pendingBytes.length >= PCM_FRAME_BYTES) {
        frameBuffer = this.pendingBytes.subarray(0, PCM_FRAME_BYTES);
        this.pendingBytes = this.pendingBytes.subarray(PCM_FRAME_BYTES);
      } else {
        frameBuffer = Buffer.alloc(PCM_FRAME_BYTES);
      }

      const frameSamples8k = cloneSamples(pcmBufferToSamples(frameBuffer));
      const frameSamples48k = upsample8kTo48k(frameSamples8k);
      this.audioSource.onData({
        bitsPerSample: 16,
        channelCount: CHANNEL_COUNT,
        numberOfFrames: WEBRTC_FRAME_SAMPLES,
        sampleRate: WEBRTC_SAMPLE_RATE,
        samples: frameSamples48k,
      });
    }, AUDIO_FRAME_MS);
  }

  stop() {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.pendingBytes = Buffer.alloc(0);
  }
}

class AudioSinkBridge {
  constructor(sendBinary) {
    this.sendBinary = sendBinary;
    this.sinks = new Map();
  }

  attachStream(stream) {
    if (!stream || typeof stream.getAudioTracks !== "function") {
      return;
    }
    for (const track of stream.getAudioTracks()) {
      this.attachTrack(track);
    }
    if (typeof stream.addEventListener === "function") {
      stream.addEventListener("addtrack", (event) => {
        if (event?.track?.kind === "audio") {
          this.attachTrack(event.track);
        }
      });
    }
  }

  attachTrack(track) {
    if (!track || this.sinks.has(track.id)) {
      return;
    }

    const sink = new wrtc.nonstandard.RTCAudioSink(track);
    sink.ondata = (audioData) => {
      try {
        const samples48k = cloneSamples(audioData.samples);
        const samples8k = downsample48kTo8k(samples48k);
        this.sendBinary(samplesToBuffer(samples8k));
      } catch (error) {
        log("error", "Failed to process remote audio", { message: String(error) });
      }
    };
    this.sinks.set(track.id, sink);
  }

  stop() {
    for (const sink of this.sinks.values()) {
      sink.stop();
    }
    this.sinks.clear();
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const assistantId = ensureRequiredArg(args, "assistant-id");
  const wsPort = Number.parseInt(ensureRequiredArg(args, "ws-port"), 10);
  const conversationId = args["conversation-id"];

  if (!Number.isInteger(wsPort) || wsPort <= 0) {
    throw new Error(`Invalid --ws-port value: ${args["ws-port"]}`);
  }

  const audioSource = new wrtc.nonstandard.RTCAudioSource();
  const localTrack = audioSource.createTrack();
  const localStream = new wrtc.MediaStream([localTrack]);
  const remoteStream = new wrtc.MediaStream();

  installBrowserShims(localStream);

  let pythonSocket = null;
  let activeCall = null;
  let readySent = false;
  let shuttingDown = false;

  const sendControl = (payload) => {
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(JSON.stringify(payload));
    }
  };

  const sendBinary = (payload) => {
    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.send(payload, { binary: true });
    }
  };

  const audioSourceBridge = new AudioSourceBridge(audioSource);
  const audioSinkBridge = new AudioSinkBridge(sendBinary);
  audioSinkBridge.attachStream(remoteStream);

  const allCodecs = wrtc.RTCRtpReceiver.getCapabilities("audio")?.codecs ?? [];
  const opusCodec = allCodecs.find((codec) => String(codec.mimeType || "").toLowerCase().includes("opus"));

  const client = new TelnyxRTC({
    anonymous_login: {
      target_id: assistantId,
      target_type: "ai_assistant",
      ...(conversationId ? { target_params: { conversation_id: conversationId } } : {}),
    },
  });

  const cleanup = async (reason = "hangup") => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;

    audioSourceBridge.stop();
    audioSinkBridge.stop();

    try {
      if (activeCall) {
        activeCall.hangup();
      }
    } catch (error) {
      log("warn", "Failed to hang up active call", { message: String(error) });
    }

    try {
      await client.disconnect();
    } catch (error) {
      log("warn", "Failed to disconnect Telnyx client", { message: String(error) });
    }

    try {
      localTrack.stop();
    } catch (error) {
      log("warn", "Failed to stop local track", { message: String(error) });
    }

    sendControl({ type: reason });

    if (pythonSocket && pythonSocket.readyState === WebSocket.OPEN) {
      pythonSocket.close();
    }
    server.close(() => {
      process.exit(0);
    });
  };

  const server = new WebSocketServer({ host: WS_HOST, port: wsPort });
  server.on("connection", (socket) => {
    pythonSocket = socket;
    log("info", "Connected to Python bridge");

    socket.on("message", (data, isBinary) => {
      if (isBinary) {
        audioSourceBridge.enqueuePcm8k(Buffer.from(data));
        return;
      }

      try {
        const message = JSON.parse(String(data));
        if (message.type === "hangup") {
          void cleanup("hangup");
        }
      } catch (error) {
        sendControl({ type: "error", message: `Invalid JSON control frame: ${String(error)}` });
      }
    });

    socket.on("close", () => {
      void cleanup("hangup");
    });
  });

  server.on("listening", () => {
    log("info", "Local WebSocket server listening", { host: WS_HOST, port: wsPort });
  });

  server.on("error", (error) => {
    log("error", "Local WebSocket server error", { message: String(error) });
    sendControl({ type: "error", message: `Local WebSocket server error: ${String(error)}` });
  });

  client.on("telnyx.ready", () => {
    log("info", "Telnyx client ready");
    activeCall = client.newCall({
      destinationNumber: "",
      localStream,
      remoteStream,
      ...(opusCodec ? { preferred_codecs: [opusCodec] } : {}),
    });
    audioSourceBridge.start();
  });

  client.on("telnyx.error", (error) => {
    const message = error?.error?.message || error?.message || "Unknown Telnyx error";
    log("error", "Telnyx client error", { message });
    sendControl({ type: "error", message });
  });

  client.on("telnyx.notification", (notification) => {
    const call = notification?.call;
    if (!call) {
      return;
    }

    activeCall = call;
    if (call.remoteStream) {
      audioSinkBridge.attachStream(call.remoteStream);
    }

    log("info", "Telnyx notification", { state: call.state, type: notification.type });

    if (!readySent && call.state === "active") {
      readySent = true;
      sendControl({ type: "ready" });
    }

    if (["hangup", "destroy", "purge"].includes(String(call.state))) {
      void cleanup("hangup");
    }
  });

  process.on("SIGINT", () => {
    void cleanup("hangup");
  });
  process.on("SIGTERM", () => {
    void cleanup("hangup");
  });

  client.connect();
}

main().catch((error) => {
  log("error", "Fatal Telnyx helper error", { message: String(error), stack: error?.stack });
  process.exit(1);
});
