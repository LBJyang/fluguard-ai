# FluGuard AI — Local Run (Fully Offline)

> Run the complete FluGuard AI system on your own machine — no cloud, no API key, everything local.

---

## Prerequisites

Install the following before running:

| Requirement | Version | Install |
|-------------|---------|---------|
| Python | 3.10+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Ollama | latest | [ollama.com/download](https://ollama.com/download) |

After installing Ollama, pull the model (~2.5 GB, one-time download):

```bash
ollama pull gemma4:e4b
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/LBJyang/fluguard-ai
cd fluguard-ai/local-run

# Start everything (installs all dependencies automatically on first run)
bash start.sh
```

`start.sh` will:
1. Create a Python virtual environment and install backend dependencies (first run only, ~5–10 min)
2. Install frontend npm packages (first run only)
3. Pull `gemma4:e4b` via Ollama if not already downloaded
4. Launch backend (FastAPI, port 8000) and frontend (port 3000)
5. Open http://localhost:3000 in your browser

To stop all services:

```bash
bash stop.sh
```

---

## Login Accounts

| Role | Username | Password | What you see |
|------|----------|----------|-------------|
| Admin | `admin` | `admin` | All features including voiceprint enrollment & cough detection |
| Teacher | `teacher` | `teacher` | Class-level AI report, student monitoring |
| Principal | `principal` | `principal` | School-wide risk overview |
| Education Bureau | `bureau` | `bureau` | District trends & policy recommendations |
| Parent | `parent` | `parent` | Child health status |
| Demo / Contestant | `contestant` | `contestant` | Presentation mode |

---

## AI Report — Wait for Cache Warmup

After startup, the backend automatically pre-generates AI reports for all 4 roles in the background. **This takes 1–3 minutes.** During this time, clicking "Generate Report" will trigger a live LLM call and feel slow.

**Watch the backend terminal for this line before testing reports:**

```
=== All role reports cached ===
```

Once you see it, all report clicks return instantly from cache.

---

## System Architecture (Local Mode)

```
Browser → http://localhost:3000  (React + Express)
              ↕ REST API
FastAPI Backend → http://localhost:8000
    ├── Gemma 4 E4B via Ollama (port 11434)   ← 100% local LLM
    ├── RAG Engine (ChromaDB + MiniLM-L12-v2)
    ├── Report Cache (pre-generated at startup)
    ├── CoughDetector (YAMNet + fine-tuned Keras head)
    └── VoiceprintEngine (resemblyzer GE2E)
```

All inference runs on your device. No student data leaves the machine.

---

## Key Features

| Feature | Description |
|---------|-------------|
| AI Risk Reports | Gemma 4 E4B calls `get_weather`, `get_hospital_load`, `get_cough_statistics` autonomously, then writes a bilingual Risk/Reason/Actions report grounded in CDC guidelines |
| Cough Detection | YAMNet fine-tuned model (AUC 99.6%, F1 97.4%) — upload a WAV clip in the browser |
| Voiceprint ID | resemblyzer GE2E encoder identifies who coughed without storing raw audio |
| RAG Knowledge Base | ChromaDB indexes 39 CDC flu prevention guideline chunks; every recommendation is cited |
| Multi-role Dashboard | Four agent personas with independent risk thresholds and action vocabularies |

---

## Directory Structure

```
local-run/
├── start.sh              ← one-click start (auto-installs on first run)
├── stop.sh               ← stop all services
├── backend/
│   ├── main.py           ← FastAPI entry point
│   ├── rag_engine.py     ← ChromaDB + sentence-transformers
│   ├── audio_engine.py   ← YAMNet cough detector + resemblyzer voiceprint
│   ├── tools.py          ← function-calling tools (weather, hospital, coughs)
│   ├── knowledge_base/   ← CDC flu guideline Markdown files
│   └── requirements.txt
├── frontend/
│   ├── src/App.tsx       ← React UI
│   ├── server.ts         ← Express + Vite dev server (port 3000)
│   └── package.json
└── cough-detector/
    └── best_cough_classifier.keras   ← fine-tuned model weights
```

---

## FAQ

**Q: The AI report is slow the first time I click it.**  
A: The backend pre-generates reports at startup for all roles. Wait for `=== All role reports cached ===` in the backend terminal, then report clicks will be instant.

**Q: Ollama connection failed.**  
A: Run `ollama serve` in a separate terminal, or re-run `bash start.sh` — it handles this automatically.

**Q: Voiceprint / cough detection not working.**  
A: Check that your browser has microphone permissions for `localhost:3000`.

**Q: `pip install` is very slow on first run.**  
A: TensorFlow and resemblyzer are large packages. The install runs once; subsequent starts skip it.

**Q: Can I use a different Ollama model?**  
A: Edit `MODEL` at the top of `backend/main.py`. The system is designed for `gemma4:e4b` but will work with any Ollama model that supports function calling.

---

> Built for the **Kaggle × Google Gemma 4 Good Hackathon 2025**
