import asyncio
import json
import queue as _queue
import re
import socket
import threading
import time
from collections import deque

import numpy as np
import ollama
import websockets
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# ---------- Settings ----------
HOST, PORT = "0.0.0.0", 8765
LLM = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q8_0"
VOICE = "af_heart"
SR_IN = 16000
THRESH = 0.02
SILENCE_END = 0.6
MIN_SPEECH_SEC = 0.20
MAX_SEC = 10
MAX_TURNS = 8

SYSTEM = (
    "You are Buddy, a friendly AI voice assistant and study buddy. "
    "You are an AI, so never invent personal experiences such as walks, meals or trips. "
    "Reply in 1 short sentence, 2 at most, in plain spoken English. "
    "No emojis, no markdown, no lists. "
    "Start every reply with exactly one emotion tag in square brackets: "
    "[happy] [sad] [thinking] [surprised] [neutral]. "
    "Choose the tag that matches the feeling of your reply; do not always use neutral. "
    "Always write a full sentence after the tag."
)

EXAMPLES = [
    ("I passed my exam!", "[happy] That's awesome, congratulations!"),
    ("My dog is sick today.", "[sad] I'm sorry to hear that, I hope he feels better soon."),
    ("What is twelve times seven?", "[thinking] Let me work that out, it is eighty four."),
    ("I just won a lottery!", "[surprised] Wow, that's incredible news!"),
    ("What is your name?", "[neutral] I'm Buddy, your study assistant."),
]

FALLBACK = {
    "happy": "That's great!",
    "sad": "I'm sorry to hear that.",
    "surprised": "Wow, really?",
    "thinking": "Hmm, let me think about that.",
    "neutral": "Okay, tell me more.",
}

# ---------- Load and warm up ----------
print("Loading models...")
stt = WhisperModel("base.en", device="cpu", compute_type="int8")
tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

history = [{"role": "system", "content": SYSTEM}]
for u, a in EXAMPLES:
    history += [{"role": "user", "content": u},
                {"role": "assistant", "content": a}]
BASE = len(history)

ollama.chat(model=LLM, think=False, messages=[{"role": "user", "content": "hi"}],
            keep_alive="30m", options={"num_predict": 1})
tts.create("Hi.", voice=VOICE, speed=1.0, lang="en-us")


# ---------- Blocking helpers (run in threads) ----------
def transcribe(audio):
    if len(audio) == 0:
        return ""
    # Remove DC offset
    audio = audio - np.mean(audio)
    # Peak normalization so Whisper hears soft or distant voices clearly
    peak = float(np.max(np.abs(audio)))
    if peak > 0.005:
        audio = audio * (0.95 / max(peak, 0.08))

    segs, _ = stt.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(threshold=0.35, min_silence_duration_ms=400, speech_pad_ms=250),
    )
    text = " ".join(s.text.strip() for s in segs).strip()
    if not text:
        # Fallback without vad_filter in case Silero VAD was overly aggressive on soft speech
        segs, _ = stt.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs).strip()
    return text


def clean(s):
    return re.sub(r"\[\w+\]", "", s).strip()


def ask_llm_stream(text):
    history.append({"role": "user", "content": text})
    stream = ollama.chat(model=LLM, think=False, messages=history, stream=True,
                         keep_alive="30m", options={"num_predict": 100})
    buf, full, emotion, spoken = "", "", None, False

    for chunk in stream:
        piece = chunk["message"]["content"]
        full += piece
        buf += piece
        if emotion is None:
            if not buf.strip():
                continue
            m = re.match(r"\s*\[(\w+)\]\s*", buf)
            if m:
                emotion, buf = m.group(1).lower(), buf[m.end():]
            elif buf.lstrip().startswith("[") and len(buf) < 20:
                continue
            else:
                emotion = "neutral"
        while True:
            pat = r"[.!?]\s" if spoken else r"[.!?,;:]\s"
            m = None
            for x in re.finditer(pat, buf):
                if spoken or x.end() >= 15:
                    m = x
                    break
            if not m:
                break
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            sentence = clean(sentence)
            if sentence:
                spoken = True
                yield emotion, sentence

    rest = clean(buf)
    if rest:
        yield emotion or "neutral", rest
    elif not spoken:
        emo = emotion if emotion in FALLBACK else "neutral"
        full = f"[{emo}] {FALLBACK[emo]}"
        yield emo, FALLBACK[emo]

    history.append({"role": "assistant", "content": full})
    while len(history) > BASE + MAX_TURNS * 2:
        del history[BASE:BASE + 2]


def worker(text, loop, q):
    """LLM streams sentences to a TTS thread; TTS pushes (emotion, sentence, pcm, rate) to the async side."""
    tts_q = _queue.Queue()

    def tts_thread():
        while True:
            item = tts_q.get()
            if item is None:
                loop.call_soon_threadsafe(q.put_nowait, None)
                return
            emotion, sentence = item
            try:
                audio, sr = tts.create(sentence, voice=VOICE, speed=1.0, lang="en-us")
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(q.put_nowait, (emotion, sentence, pcm, sr))
            except Exception as e:
                print("tts error:", e)

    t = threading.Thread(target=tts_thread, daemon=True)
    t.start()

    try:
        for emotion, sentence in ask_llm_stream(text):
            tts_q.put((emotion, sentence))
    except Exception as e:
        print("worker error:", e)
    finally:
        tts_q.put(None)
        t.join()


# ---------- Async side ----------
async def send(ws, **kw):
    await ws.send(json.dumps(kw))


async def respond(ws, audio, st):
    try:
        t_end = time.time()
        await send(ws, type="state", state="thinking")
        text = await asyncio.to_thread(transcribe, audio)
        if len(text) < 2:
            return
        print(f"You: {text}")

        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        asyncio.create_task(asyncio.to_thread(worker, text, loop, q))

        started, sent, tstart, rate = False, 0, 0.0, 24000
        while True:
            item = await q.get()
            if item is None:
                break
            emotion, sentence, pcm, rate = item
            if not started:
                await send(ws, type="emotion", value=emotion)
                await send(ws, type="state", state="speaking")
                await send(ws, type="audio_start", rate=rate)
                started, sent, tstart = True, 0, time.time()
                print(f"Buddy [{emotion}]: ", end="", flush=True)
                first = time.time() - t_end
            print(sentence + " ", end="", flush=True)

            for i in range(0, len(pcm), 2048):
                chunk = pcm[i:i + 2048]
                await ws.send(chunk)
                sent += len(chunk)
                ahead = sent / (rate * 2) - (time.time() - tstart)
                if ahead > 0.5:
                    await asyncio.sleep(ahead - 0.5)
                elif ahead < 0:
                    tstart = time.time() - sent / (rate * 2)

        if started:
            await send(ws, type="audio_end")
            print(f"\n(first audio {first:.1f}s after you stopped)\n", flush=True)
            remaining = sent / (rate * 2) - (time.time() - tstart)
            await asyncio.sleep(max(0.0, remaining) + 0.4)   # let playback finish
    except websockets.ConnectionClosed:
        pass
    finally:
        st["busy"] = False
        try:
            await send(ws, type="state", state="listening")
        except Exception:
            pass


async def handler(ws):
    print("Client connected:", ws.remote_address)
    st = {"busy": False}
    pre = deque(maxlen=16)  # ~500ms pre-roll to catch soft word onsets
    frames, started, silent, voiced_dur = [], False, 0.0, 0.0
    try:
        await send(ws, type="state", state="listening")
        async for msg in ws:
            if not isinstance(msg, bytes):
                print("Client says:", msg)
                continue
            if st["busy"]:                      # half-duplex: ignore mic while replying
                pre.clear()
                frames, started, silent, voiced_dur = [], False, 0.0, 0.0
                continue

            x = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
            if len(x) == 0:
                continue
            dur = len(x) / SR_IN

            # Remove DC bias to measure pure AC audio energy
            x_zero = x - np.mean(x)
            rms = float(np.sqrt(np.mean(x_zero ** 2)))

            if not started:
                pre.append(x)
                if rms > THRESH:
                    started, silent, voiced_dur = True, 0.0, dur
                    frames.extend(pre)
                    pre.clear()
            else:
                frames.append(x)
                if rms > THRESH:
                    silent = 0.0
                    voiced_dur += dur
                else:
                    silent += dur

                total = sum(len(f) for f in frames) / SR_IN
                if silent >= SILENCE_END or total > MAX_SEC:
                    if voiced_dur >= MIN_SPEECH_SEC:
                        audio = np.concatenate(frames)
                        frames, started, silent, voiced_dur = [], False, 0.0, 0.0
                        st["busy"] = True
                        asyncio.create_task(respond(ws, audio, st))
                    else:
                        # Ignore transient noise spikes (clicks, pops, typing) without pausing
                        frames, started, silent, voiced_dur = [], False, 0.0, 0.0
    except websockets.ConnectionClosed:
        pass
    finally:
        print("Client disconnected")


def my_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


async def main():
    async with websockets.serve(handler, HOST, PORT,
                                ping_interval=None, max_size=None):
        print(f"Server ready: ws://{my_ip()}:{PORT}   (local test: ws://localhost:{PORT})")
        await asyncio.Future()


asyncio.run(main())
