# FluGuard AI — School Flu Monitoring Powered by Gemma 4

> **AI-powered early warning system that detects classroom flu outbreaks hours before the first absence report — using on-device audio AI, RAG-grounded medical knowledge, and Gemma 4's native agentic function calling.**

[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY%204.0-lightgrey.svg)](LICENSE)

---

## The Problem

When my son started primary school, I watched the same quiet pattern every winter: kids coughing for days before absences began. By the time the school noticed, half the class was gone. He caught the flu in his first year, aged six. It became pneumonia. He was hospitalised for two weeks.

**Flu doesn't start with a fever. It starts with a cough.** Schools have no system to catch that signal early. FluGuard AI changes that.

---

## What It Does

FluGuard AI is a real-time flu surveillance platform for schools:

- **YAMNet Cough Detection** — classroom microphones run a fine-tuned audio classifier (AUC 99.6%) that counts and triages coughs per student, continuously
- **Voiceprint Identification** — resemblyzer GE2E encoder identifies *who* coughed without storing raw audio
- **Gemma 4 Agentic Reports** — four AI agent personas (teacher, principal, education bureau, parent) each autonomously call `get_weather()`, `get_hospital_load()`, and `get_cough_statistics()`, then synthesise a bilingual Risk / Reason / Action report grounded in CDC guidelines
- **RAG-Grounded Medical Knowledge** — ChromaDB indexes 39 chunks of CDC flu prevention guidelines; every recommendation cites the retrieved chunk

---

## Demo

### Screenshots

> _[SCREENSHOT_1: Dashboard with cough trend charts and risk level badge]_
> _[SCREENSHOT_2: Role-based AI report showing Risk/Reason/Actions]_
> _[SCREENSHOT_3: Live cough detection in-browser with probability readout]_
> _[SCREENSHOT_4: Voiceprint enrollment and verification UI]_

### Video Demo

**[VIDEO_LINK]** — 3-minute walkthrough of the full system

### Live Demo

**[DEMO_LINK]** — Try it without any setup (note: AI features require the backend to be running)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User's Browser                           │
│  React 19 + TypeScript + Vite                                   │
│  • Role dashboard (teacher / principal / bureau / parent)       │
│  • Web Audio API → in-browser WAV encoding (no ffmpeg needed)   │
│  • Recharts cough trend visualisation                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTPS REST API
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Railway Backend (FastAPI)                      │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  RAG Engine     │  │  Audio Engine    │  │  Tool Engine  │  │
│  │  ChromaDB       │  │  YAMNet (TF Hub) │  │  get_weather  │  │
│  │  MiniLM-L12-v2  │  │  + keras head    │  │  get_hospital │  │
│  │  39 CDC chunks  │  │  resemblyzer GE2E│  │  get_coughs   │  │
│  └────────┬────────┘  └──────────────────┘  └───────┬───────┘  │
│           │                                          │          │
│           └──────────────────┬───────────────────────┘          │
│                              ▼                                  │
│                    Agentic Loop (max 2 rounds)                  │
│                    builds messages → calls tools → synthesises  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTPS
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Google AI Studio API                               │
│              Model: gemma-4-27b-it (Gemma 4 MoE)               │
│              Native function calling enabled                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## How We Use Gemma 4

### Model

**`gemma-4-27b-it`** via Google AI Studio API — the Gemma 4 mixture-of-experts instruction-tuned model (27B total parameters, ~4B active per forward pass). Chosen because:

1. **Native function calling** works out of the box — no prompt engineering workarounds needed
2. **Multilingual** (Chinese + English) without fine-tuning, critical for a Chinese school setting
3. **Cost-effective at inference** — MoE activation means lower latency and API cost vs. a dense 27B model

### How It's Called

The backend uses **Google AI Studio's `google-genai` Python SDK** for all LLM calls:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemma-4-27b-it",
    contents=contents,            # converted from OpenAI-format message history
    config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2048,
        system_instruction=system_prompt,
        tools=[types.Tool(function_declarations=declarations)],
    ),
)
```

### Agentic Function Calling Loop

Three tools are registered with Gemma 4's native function-calling interface:

| Tool | Data Source | Purpose |
|------|-------------|---------|
| `get_weather(city)` | Open-Meteo free API (live) | Temperature, humidity → flu transmission risk |
| `get_hospital_load()` | Simulated district health bureau feed | Pediatric flu patient load at nearby hospitals |
| `get_cough_statistics(class_id)` | YAMNet classroom monitoring | Per-class cough counts, trends, risk levels |

When a report is triggered, Gemma 4 **autonomously plans its reasoning chain**: it issues tool calls sequentially, receives structured results, and composes the final bilingual Risk Assessment grounded in real, timestamped data.

**Safety-critical fields (Risk Level, Data Used) are assembled deterministically from tool return values** — not generated by the LLM. This prevents hallucination on the most consequential parts of a medical report.

### RAG Grounding

ChromaDB indexes 39 paragraphs of CDC flu prevention guidelines encoded with `paraphrase-multilingual-MiniLM-L12-v2`. Every Gemma 4 call receives the top-3 relevant chunks in its system prompt. Every recommendation traces to a specific guideline — the model cannot invent medical advice.

### Role-Specific Multi-Agent Design

Four distinct Gemma 4 agent personas, each with different risk thresholds and action vocabularies:

- **Teacher** — individual student triage, ventilation schedules, parent notification
- **Principal** — school-wide metrics, class suspension decisions, bureau reporting
- **Education Bureau** — district-level trends, resource allocation, policy actions
- **Parent** — child-specific guidance, hospital navigation, home care

---

## YAMNet Cough Detector — Technical Details

Fine-tuned on a real cough dataset using **domain adaptation**:

- **Backbone**: YAMNet (TF Hub, frozen) → 1024-dim frame embeddings
- **Head**: Dense classification head, trained on labelled cough/non-cough audio
- **Performance**: AUC 99.6%, F1 97.4%, Recall 99.33%, Precision 96.75%
- **Latency**: < 200ms per clip, runs entirely on server CPU
- **Privacy**: Speaker identity stored as 256-dim d-vector (resemblyzer GE2E), never raw audio

---

## Training Notebooks

The `notebook/` directory contains the full model training code, runnable on Kaggle:

| Notebook | Description |
|----------|-------------|
| [`nb-yamnet-ft-v1.ipynb`](notebook/nb-yamnet-ft-v1.ipynb) | Baseline: YAMNet domain adaptation on CoughVid v3 + ESC-50. Achieves AUC 99.0%. |
| [`nb-yamnet-ft-v2.ipynb`](notebook/nb-yamnet-ft-v2.ipynb) | **Deployed model**: Adds LibriSpeech speech negatives to fix speech→cough misclassification. Final AUC **99.6%**, F1 **97.4%**. |

Both notebooks are self-contained and reproducible on a Kaggle GPU (T4, ~5 min). They demonstrate the full ML pipeline: data assembly → YAMNet embedding extraction → classifier training → evaluation.

---

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google AI Studio API key ([get one free](https://aistudio.google.com/apikey))

### Backend

```bash
cd deploy/backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd deploy/frontend
npm install
cp .env.example .env
# Edit .env and set VITE_BACKEND_URL=http://localhost:8000
npm run dev
```

Open http://localhost:5173

---

## Team

> _[TEAM_MEMBER_1] — Role_
> _[TEAM_MEMBER_2] — Role_

---

## Tracks Targeted

| Track | Qualification |
|-------|--------------|
| **Health & Sciences** | Democratizes hospital-grade flu intelligence for any school — no medical staff needed |
| **Global Resilience** | Works offline after first model load; designed for schools with no IT budget |
| **Safety & Trust** | Traceable citations, deterministic safety fields, RAG-grounded outputs, no hallucinated medical advice |

---

## Open Source License

This project is released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)** license, as required by the Gemma 4 Good Hackathon.

See [LICENSE](LICENSE) for the full text.
