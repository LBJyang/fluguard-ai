# FluGuard AI — Deployment Guide

Complete step-by-step instructions for deploying FluGuard AI to Railway (backend) and Vercel (frontend).

---

## Environment Variables Reference

### Backend (Railway)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | **Yes** | Google AI Studio API key. Get one free at https://aistudio.google.com/apikey |
| `GEMMA_MODEL` | No | Override the Gemma 4 model ID. Default: `gemma-4-27b-it` |
| `PORT` | No | Railway sets this automatically. Do not override. |

### Frontend (Vercel)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_BACKEND_URL` | **Yes** | Full URL of your Railway backend, e.g. `https://fluguard-ai.up.railway.app` (no trailing slash) |

---

## 1. Deploy Backend to Railway

### Prerequisites
- Railway account (https://railway.app) — free tier works
- Git repository containing this code pushed to GitHub

### Steps

1. **Create a new Railway project**
   - Go to https://railway.app/new
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account and select this repository

2. **Set the root directory to `deploy/backend`**
   - In Railway project settings → Source → Root Directory: `deploy/backend`

3. **Add environment variables**
   - Go to project → Variables tab
   - Add: `GEMINI_API_KEY` = your Google AI Studio key
   - (Optional) Add: `GEMMA_MODEL` = `gemma-4-27b-it`

4. **Configure service settings**
   - Railway will auto-detect Python and use `railway.toml` for start command
   - Start command (from `railway.toml`): `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/ping`

5. **Deploy**
   - Railway deploys automatically on every push to your main branch
   - First deploy takes ~5-10 minutes (downloads sentence-transformers model ~420 MB)
   - Check logs: Railway dashboard → Deployments → View Logs

6. **Note your backend URL**
   - After deploy succeeds, Railway shows a public URL like `https://fluguard-abc123.up.railway.app`
   - Test: `curl https://your-url.up.railway.app/ping` should return `"ok"`

### Railway Memory Requirements

> ⚠️ The cough detector uses TensorFlow. The free Railway plan (512 MB RAM) may be insufficient.
> Upgrade to the **Hobby plan** (8 GB RAM) if the backend crashes on startup.
> Alternatively, set `DISABLE_AUDIO=true` in the future if audio features are not needed for your demo.

---

## 2. Deploy Frontend to Vercel

### Prerequisites
- Vercel account (https://vercel.com) — free tier works
- Backend deployed and URL noted from Step 1

### Steps

1. **Import project to Vercel**
   - Go to https://vercel.com/new
   - Import from GitHub → select this repository

2. **Set the root directory to `deploy/frontend`**
   - In the "Configure Project" screen → Root Directory: `deploy/frontend`
   - Framework Preset: Vite (auto-detected)

3. **Add environment variables**
   - Expand "Environment Variables"
   - Add: `VITE_BACKEND_URL` = `https://your-railway-backend-url.up.railway.app`
     (use the URL from Railway Step 6, no trailing slash)

4. **Deploy**
   - Click "Deploy"
   - Build takes ~2 minutes
   - Vercel shows your live URL: `https://fluguard-ai.vercel.app`

5. **Test the deployment**
   - Open the Vercel URL
   - Select a role (e.g. Principal)
   - Click "Generate Report" — the report should load within 10-30 seconds
   - Try the chat interface

---

## 3. Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Google AI Studio API key

### Backend (local)

```bash
# Navigate to backend directory
cd deploy/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here

# Start the server
uvicorn main:app --reload --port 8000
```

The backend will be available at http://localhost:8000

Verify it's running:
```bash
curl http://localhost:8000/ping
# Expected: "ok"

curl http://localhost:8000/api/health
# Expected: JSON with status, rag_chunks, etc.
```

### Frontend (local)

```bash
# In a new terminal, navigate to frontend directory
cd deploy/frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# The .env file should have: VITE_BACKEND_URL=http://localhost:8000

# Start dev server
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 4. Common Issues & Troubleshooting

### Backend won't start — "GEMINI_API_KEY is not set"
- Make sure you added the env var in Railway (Variables tab) or in your local `.env` file
- The key must start with `AIza...`

### Backend crashes with "Cannot allocate memory"
- TensorFlow requires ~1.5 GB RAM for the cough detector
- On Railway: upgrade to Hobby plan (8 GB RAM)
- Alternatively, the API endpoints that don't use audio (`/api/chat`, `/api/report`) still work even if the audio engine fails to load

### Frontend shows "FluGuard backend error 503"
- Backend may still be starting up (first startup takes ~2 minutes while models download)
- Check Railway logs for startup progress
- Verify `VITE_BACKEND_URL` in Vercel is correct and has no trailing slash

### Report generation returns fallback text instead of AI-generated text
- This is expected behaviour — fallbacks are bilingual, data-grounded, and production-quality
- If you want to verify live AI generation, check the backend logs for `Google AI Studio call: summary`

### CORS errors in browser console
- Backend allows `*` origins by default — this should not happen
- If it does, check that `VITE_BACKEND_URL` points to the Railway URL (not localhost)

### Frontend build fails — TypeScript errors
- Run `npm run lint` to check for type errors
- The `src/ml/engine.ts` file uses browser-only TensorFlow.js — it is not used server-side

### Cough detection returns "CoughDetector not ready yet"
- The audio engine loads asynchronously in the background at startup
- Wait 60-90 seconds after backend startup for the cough detector to initialise
- Check logs for "CoughDetector ready ✓"

---

## 5. Verifying the Full Stack

After both services are deployed, run through this checklist:

- [ ] `GET /ping` → `"ok"`
- [ ] `GET /api/health` → JSON with `google_ai_studio: true`
- [ ] Frontend loads at Vercel URL
- [ ] Role selection works (Teacher / Principal / Bureau / Parent)
- [ ] "Generate Report" returns a risk assessment with Reason and Actions
- [ ] Chat interface responds to a message
- [ ] (Optional) Cough detection accepts an audio file upload
- [ ] (Optional) Voiceprint enrollment and verification work
