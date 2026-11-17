<div align="center">

# EchoMind

**A fully local voice AI assistant on a plain ESP32.**
Mic in, local STT + LLM + TTS on your laptop, speaker and OLED emotion face out. No cloud, no API keys.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32-Arduino-E7352C?logo=espressif&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000)
![STT](https://img.shields.io/badge/STT-faster--whisper-blue)
![TTS](https://img.shields.io/badge/TTS-Kokoro-purple)
![Platform](https://img.shields.io/badge/Laptop-Windows-0078D6?logo=windows&logoColor=white)

</div>

---

## Features

- **100% local and private:** speech-to-text, the LLM and text-to-speech all run on your laptop.
- **Uncensored local LLM:** runs [Qwen3.5-4B Uncensored (HauhauCS)](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive) through Ollama. Swap in any Ollama model.
- **Pick a model for your GPU:** quantizations from 2.6 GB to 7.9 GB, see [Choose your model](#2-choose-your-model).
- **Emotion display:** the OLED face changes with the reply (happy, sad, thinking, surprised, neutral).
- **Fast replies:** the answer is spoken sentence by sentence while the rest is still generating.
- **Test without hardware:** `fake_esp32.py` simulates the ESP32 so you can try the whole pipeline on the laptop.

## How it works

```
ESP32 (INMP441 mic)
      |  audio over WebSocket (Wi-Fi)
      v
Laptop server (Python)
  faster-whisper (STT) -> Ollama LLM -> Kokoro (TTS)
      |  reply audio + emotion over WebSocket
      v
ESP32 (MAX98357A amp + speaker, OLED emotion face)
```

| Stage | Tool | Runs on |
|---|---|---|
| Microphone capture | INMP441 (I2S) | ESP32 |
| Speech-to-text | faster-whisper (`base.en`) | Laptop |
| Language model | Ollama (default: Qwen3.5-4B Uncensored, Q8_0) | Laptop |
| Text-to-speech | Kokoro (`af_heart` voice) | Laptop |
| Audio playback | MAX98357A + speaker | ESP32 |
| Emotion display | 0.96" OLED (SSD1306, 128x64) | ESP32 |

## Project status

| Part | Status |
|---|---|
| Laptop voice loop (STT -> LLM -> TTS) | Working |
| OLED emotion display | Working |
| ESP32 microphone streaming | In progress |
| Wake word | Planned |
| ESP32 firmware | I2S / OLED hardware test only |

## Quick start (Windows, PowerShell)

### 1. Install Ollama

```powershell
winget install Ollama.Ollama
```

No winget? Download it from https://ollama.com/download. **Restart PowerShell afterwards**, then check with `ollama --version`.

### 2. Choose your model

The default is **Q8_0**. Pick the file that fits your GPU, then pull it (the download shows a progress bar):

| Your GPU VRAM | Pick | Size | Notes |
|---|---|---|---|
| CPU only or 2-4 GB | `Q4_K_M` | 2.6 GB | Fastest, lowest memory use |
| 4-6 GB | `Q6_K` | 3.3 GB | Good balance |
| 6-8 GB (RTX 4050 class) | `Q8_0` | 4.2 GB | Best quality that fits in 6 GB. Default. |
| 12 GB+ | `BF16` | 7.9 GB | Full precision, rarely worth it for voice |

Sizes are file sizes. Leave some VRAM free for context. For voice, a smaller quant means faster replies. Other sizes (like the 9B) are on the [HauhauCS Hugging Face page](https://huggingface.co/HauhauCS).

```powershell
ollama pull hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q8_0
```

Swap `Q8_0` for your pick. If you choose a different quant, set the same name in `LLM = "..."` at the top of `server/server.py` and `server/buddy.py`.

> The model also ships a vision file (`mmproj`). Echo is text-only, so you do not need it.

### 3. Clone and install dependencies

```powershell
git clone https://github.com/PRITESH-95/ECHO.git
cd ECHO\server
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -U ollama
```

### 4. Download the Kokoro voice files (once)

```powershell
curl.exe -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl.exe -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 5. Start the server

Keep Ollama running in the background, then:

```powershell
python server.py
```

The first start downloads the Whisper model (~145 MB) and loads everything, so it takes a while. When you see `Server ready: ws://<your-ip>:8765`, it is up.

### 6. Test without hardware

In a second PowerShell window:

```powershell
cd ECHO\server; python fake_esp32.py
```

## ESP32 setup

> The `firmware` folder currently holds an I2S audio and OLED hardware test. Wi-Fi/WebSocket streaming from the ESP32 is still in progress.

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) and add ESP32 board support (Boards Manager, "esp32" by Espressif).
2. Install `U8g2` and a WebSocket client library such as `WebSockets` by Markus Sattler.
3. Put your Wi-Fi details and the laptop address (`ws://<your-ip>:8765`) in a local `config.h`. **Never commit it.**
4. Select **ESP32 Dev Module**, pick the COM port, and upload. If it stalls at "Connecting...", hold **BOOT**.
5. Laptop and ESP32 must be on the same Wi-Fi network.

### Hardware

- ESP32 dev module (plain ESP32, not S3)
- INMP441 I2S microphone
- MAX98357A I2S amplifier + speaker
- 0.96" I2C OLED (128x64)
- Breadboard, jumper wires, USB cable

### Wiring

Example pins. Match them to the pin definitions in your firmware.

| Module pin | ESP32 pin |
|---|---|
| INMP441 VDD / GND | 3V3 / GND |
| INMP441 SCK | GPIO 14 |
| INMP441 WS | GPIO 15 |
| INMP441 SD | GPIO 32 |
| INMP441 L/R | GND |
| MAX98357A VIN / GND | 5V (VIN) / GND |
| MAX98357A BCLK | GPIO 26 |
| MAX98357A LRC | GPIO 25 |
| MAX98357A DIN | GPIO 27 |
| OLED VCC / GND | 3V3 / GND |
| OLED SDA | GPIO 21 |
| OLED SCL | GPIO 22 |

## Configuration

Settings are at the top of `server/server.py`:

| Setting | What it does |
|---|---|
| `LLM` | Ollama model name |
| `VOICE` | Kokoro voice (default `af_heart`) |
| `THRESH` | Mic loudness needed to count as speech. Lower it if Echo cannot hear you. |
| `SILENCE_END` | Seconds of silence before your sentence is considered finished |
| `SYSTEM` | Personality and reply length. Edit this to change how Echo talks. |

The code calls the model with `think=False` to skip the model's long reasoning step, which keeps voice replies fast. This needs the `ollama` Python package 0.6 or newer.

## Project structure

```
ECHO/
+-- firmware/
|   +-- echo_firmware/
|       +-- echo_firmware.ino   # ESP32 I2S + OLED test
+-- server/
|   +-- server.py               # WebSocket server: STT -> LLM -> TTS
|   +-- buddy.py                # Terminal version with timing output
|   +-- fake_esp32.py           # Simulated ESP32 for testing
|   +-- chat_test.py            # LLM-only test
|   +-- stt_test.py             # Speech-to-text test
|   +-- tts_test.py             # Text-to-speech test
|   +-- kokoro_test.py          # Kokoro voice test
|   +-- requirements.txt
+-- .gitignore
+-- LICENSE
+-- README.md
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `only one usage of each socket address` on start | An old server is still running. Free the port: `Get-NetTCPConnection -LocalPort 8765 -EA 0 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| Echo does not hear me | Lower `THRESH` in `server.py` (try `0.005`) and speak close to the mic |
| Replies are too short | Edit the reply-length rule and the examples in `SYSTEM` / `EXAMPLES` |
| Replies are slow | Use a smaller quant (`Q4_K_M`) and close heavy programs |
| Whisper download is stuck | `$env:HF_HUB_DISABLE_XET="1"` then run the server again |
| `think` argument error | `pip install -U ollama` |
| `ollama` not found | Restart PowerShell after installing Ollama |
| ESP32 cannot connect | Same Wi-Fi, correct laptop IP, and allow Python through Windows Firewall (private networks) |
| Mic returns zeros or noise | INMP441 L/R must go to GND. Check SCK, WS and SD wiring. |
| No sound from speaker | Power the MAX98357A from 5V and check the speaker wires |
| OLED is blank | Check the I2C address (usually `0x3C`) and the SDA/SCL pins |

## Roadmap

- [x] Local STT -> LLM -> TTS loop
- [x] OLED emotion display
- [ ] ESP32 Wi-Fi + WebSocket audio streaming
- [ ] Wake word detection
- [ ] Interrupt a reply by speaking

## Security

Never commit `config.h`. It holds your Wi-Fi password. `.gitignore` already excludes it.

## Credits

- Built by [PRITESH-95](https://github.com/PRITESH-95).
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx), [Ollama](https://ollama.com), [HauhauCS Qwen3.5 4B](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive).

## License

See [LICENSE](LICENSE).
