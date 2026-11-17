import re
import ollama

MODEL = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q8_0"

SYSTEM = (
    "You are Buddy, a friendly voice assistant. "
    "Reply in 1 to 2 short sentences. No emojis, no markdown, no lists. "
    "Start every reply with exactly one emotion tag in square brackets, "
    "chosen from: [happy] [sad] [thinking] [surprised] [neutral]."
)

history = [{"role": "system", "content": SYSTEM}]

while True:
    q = input("You: ")
    if q.strip().lower() == "exit":
        break

    history.append({"role": "user", "content": q})
    r = ollama.chat(model=MODEL, think=False, messages=history)
    text = r["message"]["content"]
    history.append({"role": "assistant", "content": text})

    m = re.match(r"\s*\[(\w+)\]\s*(.*)", text, re.S)
    emotion, reply = (m.group(1).lower(), m.group(2)) if m else ("neutral", text)

    print(f"[emotion={emotion}] {reply}\n")
