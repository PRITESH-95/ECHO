import re
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import ollama
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# ---------- Settings ----------
LLM = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q8_0"
VOICE = "af_heart"
SR = 16000
BLOCK = int(0.1 * SR)
THRESH = 0.015
SILENCE_END = 0.7
MAX_SEC = 15
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
BASE = len(history)                      # examples stay; only real turns get trimmed

ollama.chat(model=LLM, think=False, messages=[{"role": "user", "content": "hi"}],
            keep_alive="30m", options={"num_predict": 1})
tts.create("Hi.", voice=VOICE, speed=1.0, lang="en-us")
print("Ready. Press Ctrl+C to quit.\n")

# ---------- Speech pipeline (background threads) ----------
q_text, q_audio = queue.Queue(), queue.Queue()
done = threading.Event()
mark = {"t_end": 0.0, "first": None, "tts1": None}


def synth_worker():
    while True:
        s = q_text.get()
        if s is None:
            q_audio.put(None)
            continue
        t = time.time()
        audio, sr = tts.create(s, voice=VOICE, speed=1.0, lang="en-us")
        if mark["tts1"] is None:
            mark["tts1"] = time.time() - t
        q_audio.put((audio, sr))


def play_worker():
    while True:
        item = q_audio.get()
        if item is None:
            done.set()
            continue
        if mark["first"] is None:
            mark["first"] = time.time() - mark["t_end"]
        audio, sr = item
        sd.play(audio, sr)
        sd.wait()


threading.Thread(target=synth_worker, daemon=True).start()
threading.Thread(target=play_worker, daemon=True).start()


# ---------- Functions ----------
def record():
    print("Listening...")
    frames, prev = [], None
    started, silent, t0 = False, 0.0, time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK) as stream:
        while True:
            data, _ = stream.read(BLOCK)
            data = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(data ** 2)))
            if not started:
                if rms > THRESH:
                    started = True
                    if prev is not None:
                        frames.append(prev)
                    frames.append(data)
                elif time.time() - t0 > 20:
                    return None
                prev = data
            else:
                frames.append(data)
                silent = silent + 0.1 if rms < THRESH else 0.0
                if silent >= SILENCE_END or len(frames) * 0.1 > MAX_SEC:
                    break
    return np.concatenate(frames)


def transcribe(audio):
    segs, _ = stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segs).strip()


def clean(s):
    return re.sub(r"\[\w+\]", "", s).strip()


def ask_llm_stream(text):
    """Yield (emotion, chunk). The first chunk may end at a comma for speed."""
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
    elif not spoken:                               # model gave only a tag
        emo = emotion if emotion in FALLBACK else "neutral"
        full = f"[{emo}] {FALLBACK[emo]}"
        yield emo, FALLBACK[emo]

    history.append({"role": "assistant", "content": full})
    while len(history) > BASE + MAX_TURNS * 2:
        del history[BASE:BASE + 2]


# ---------- Main loop ----------
try:
    while True:
        audio = record()
        if audio is None:
            continue
        mark["t_end"] = time.time()

        t = time.time()
        text = transcribe(audio)
        t_stt = time.time() - t
        if len(text) < 2:
            continue
        print(f"You: {text}")

        done.clear()
        mark["first"] = None
        mark["tts1"] = None
        t_llm1 = None
        t0 = time.time()
        printed = False
        for emotion, sentence in ask_llm_stream(text):
            if t_llm1 is None:
                t_llm1 = time.time() - t0
            if not printed:
                print(f"Buddy [{emotion}]: ", end="", flush=True)
                printed = True
            print(sentence + " ", end="", flush=True)
            q_text.put(sentence)
        q_text.put(None)
        done.wait()

        if mark["first"] is not None:
            print(f"\n(STT {t_stt:.1f}s | LLM first chunk {t_llm1:.1f}s | "
                  f"TTS first chunk {mark['tts1']:.1f}s | "
                  f"first audio {mark['first']:.1f}s)\n")

        if "goodbye" in text.lower():
            break
except KeyboardInterrupt:
    print("\nBye.")
