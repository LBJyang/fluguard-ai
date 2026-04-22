# FluGuard AI — Local Run (Offline Version)

This directory contains the **original, fully offline version** of FluGuard AI —
the one designed to run on a single machine with zero cloud dependency.

> This is the version described in the project writeup:
> _"Runs on a single Apple M4 Mac Mini (16 GB). No GPU required. No internet after first model pull."_

---

## Architecture (Local)

```
Browser → FastAPI (port 8000) → Ollama (port 11434) → Gemma 4 E4B (local)
                              ↳ ChromaDB + MiniLM   (local RAG)
                              ↳ YAMNet + Keras head  (local cough detection)
                              ↳ resemblyzer GE2E     (local voiceprint)
```

No API keys. No cloud. No data leaving the machine.

---

## Prerequisites

- **macOS** (start.sh uses `osascript` to open Terminal windows)
- **Ollama** — https://ollama.com → install, then run: `ollama pull gemma4:e4b`
- **Python 3.11+**
- **Node.js 18+**

---

## One-Time Setup

```bash
# 1. Backend dependencies
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # ~5-10 min first time (downloads TF + sentence-transformers)

# 2. Frontend dependencies
cd ../frontend
npm install
```

---

## Start & Stop

```bash
# Start everything (opens two Terminal windows)
bash start.sh

# Stop everything
bash stop.sh
```

After starting, open http://localhost:5173 in your browser.

---

## What start.sh Does

1. Checks if port 8000 is free
2. Starts Ollama with `gemma4:e4b` if not already running
3. Activates Python venv and starts FastAPI backend in a new Terminal window
4. Waits for backend health check to pass
5. Starts Vite dev server in a new Terminal window
6. Opens the browser automatically

---

## Differences from the Cloud Version (`../backend` + `../frontend`)

| Feature | Local (this folder) | Cloud (Vercel + Railway) |
|---------|--------------------|-----------------------|
| LLM | Ollama + Gemma 4 E4B (local) | Google AI Studio API |
| RAG | ChromaDB + sentence-transformers | scikit-learn TF-IDF |
| Cough detection | YAMNet + TensorFlow ✅ | Disabled (image size limit) |
| Voiceprint | resemblyzer ✅ | Disabled (image size limit) |
| Internet required | No (after first pull) | Yes |
| API key required | No | Yes (GEMINI_API_KEY) |
