# FluGuard AI — 项目启动指南 · Project Setup Guide
# 流感卫士 AI — Backend & Frontend Documentation

---

## ⚡ 一键启动（推荐）· Quick Start (Recommended)

项目根目录提供两个脚本，一键完成全部启动/停止操作。  
Two convenience scripts in the project root handle everything for you.

### 启动 · Start

```bash
bash start.sh
```

脚本会自动完成以下步骤 / The script will:

1. 关停 8000 端口的旧进程（若存在）/ Kill any existing process on port 8000
2. 检查 Ollama 是否运行，否则自动启动 Gemma 4 E4B / Check & start Ollama if needed
3. 在新终端窗口中启动 FastAPI 后端 / Launch FastAPI backend in a new Terminal window
4. 等待后端就绪后，启动 Vite 前端 / Wait for backend health-check, then start Vite frontend
5. 自动打开浏览器 → http://localhost:5173 / Auto-open browser

> **前提 / Prerequisites：** 需已激活虚拟环境路径 `/Users/lbjyang/ai-env`，且已安装 ffmpeg（见下方依赖说明）。  
> The script sources `/Users/lbjyang/ai-env/bin/activate`. Adjust the `VENV` variable in `start.sh` if your venv is elsewhere.

### 停止 · Stop

```bash
bash stop.sh
```

关停后端（port 8000）及前端 Vite（ports 5173 / 3000 / 4173）所有相关进程。  
Kills all backend and frontend processes (ports 8000, 5173, 3000, 4173).

---

## 架构说明 · Architecture Overview

```
Browser (React + Vite, port 5173)
    ↕  REST API (CORS: *)
FastAPI Backend (port 8000)
    ├── RAG Engine
    │   ├── ChromaDB (in-memory)
    │   └── paraphrase-multilingual-MiniLM-L12-v2 embeddings  ← 100% local
    ├── Function Calling Tools
    │   ├── get_weather()          → Open-Meteo API (free, no key needed)
    │   ├── get_hospital_load()    → simulated real-time data
    │   └── get_cough_statistics() → local audio monitoring stats
    ├── Report Cache (in-memory)
    │   └── Pre-generated at startup for all 4 roles, refreshed 08:00/12:00/16:00
    ├── CoughDetector
    │   ├── YAMNet (TF Hub, frozen)  → 1024-dim embeddings
    │   └── best_cough_classifier.keras (fine-tuned Dense head)
    ├── VoiceprintEngine
    │   └── resemblyzer GE2E speaker encoder  ← requires ffmpeg
    └── Ollama Client
            ↕  HTTP
        Ollama (port 11434)
            └── gemma4:e4b  ← 100% local, no cloud, no API key
```

**隐私保证 / Privacy guarantee:** 学生健康数据全程不离开本地设备。  
All student health data stays on-device — no cloud calls, no telemetry.

---

## 环境依赖 · Prerequisites

| 组件 / Component | 版本 / Version | 说明 / Notes |
|---|---|---|
| Python | ≥ 3.10 | 推荐 pyenv 或 conda |
| Ollama | latest | https://ollama.com |
| Node.js | ≥ 18 | 用于 React 前端 / for React frontend |
| **ffmpeg** | any | **声纹录入必须安装 / Required for voiceprint enrollment** |

### 安装 ffmpeg（声纹功能必须）· Install ffmpeg (required for voiceprint)

浏览器通过 MediaRecorder 录制的音频格式为 WebM/Opus，librosa 解码该格式需要 ffmpeg。  
Browser audio recorded via MediaRecorder uses WebM/Opus format; librosa requires ffmpeg to decode it.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

> ⚠️ **未安装 ffmpeg 会导致声纹录入报 HTTP 422 错误。**  
> Not installing ffmpeg will cause HTTP 422 errors on voiceprint enrollment.

---

## 首次安装 · First-Time Setup

### 1. 安装 Ollama 并拉取模型 · Install Ollama & Pull Model

```bash
# 从 https://ollama.com 下载安装 Ollama / Download Ollama from https://ollama.com

# 拉取 Gemma 4 E4B-IT（约 2.5 GB）/ Pull model (~2.5 GB)
ollama pull gemma4:e4b

# 验证 / Verify
ollama run gemma4:e4b "Hello"
```

### 2. 创建 Python 虚拟环境 · Set Up Python Backend

```bash
cd backend

# 创建虚拟环境 / Create venv
python -m venv venv

# 激活（macOS / Linux）/ Activate
source venv/bin/activate
# 激活（Windows）/ Activate (Windows)
venv\Scripts\activate

# 安装依赖 / Install dependencies
pip install -r requirements.txt
# sentence-transformers 首次运行会下载 ~420 MB 嵌入模型，仅下载一次
# sentence-transformers downloads ~420 MB embedding model on first run (cached)
```

### 3. 安装前端依赖 · Install Frontend Dependencies

```bash
cd 流感卫士-(fluguard-ai)
npm install
```

---

## 手动启动（备选）· Manual Start (Alternative)

如果不使用 `start.sh`，可以手动分两个终端窗口启动：  
If you prefer not to use `start.sh`, open two terminal windows:

**终端 1 — 后端 / Terminal 1 — Backend**
```bash
source /path/to/your/venv/bin/activate
cd backend
python main.py
# 预期输出 / Expected:
# INFO │ RAG engine ready — 39 chunks indexed
# INFO │ Report cache: generating for all 4 roles...
# INFO: Uvicorn running on http://0.0.0.0:8000
```

**终端 2 — 前端 / Terminal 2 — Frontend**
```bash
cd 流感卫士-(fluguard-ai)
npm run dev
# 访问 / Open: http://localhost:5173
```

---

## API 接口说明 · API Reference

### POST `/api/chat`
角色专属 AI 顾问对话（RAG + Function Calling）。  
Role-specific AI advisor chat using RAG and function calling.

```json
// Request
{
  "role": "teacher",
  "message": "今天班里咳嗽很多，应该怎么办？",
  "city": "Shenyang"
}

// Response
{
  "content": "根据当前天气（-2°C，湿度35%）...",
  "rag_sources": ["02_school_prevention_cn.md"],
  "tools_used": ["get_weather", "get_cough_statistics"]
}
```

### POST `/api/report`
触发完整 AI 报告生成（会调用 Ollama，较慢）。  
Trigger full AI report generation (calls Ollama, may take 30–60s).

```json
// Request
{ "role": "teacher" }

// Response
{
  "summary": "今日全校流感风险中等偏高...",
  "prediction": "未来24小时新增病例预计...",
  "actions": ["立即开窗通风", "..."],
  "riskScore": 72.0,
  "tools_used": ["get_weather", "get_hospital_load"]
}
```

### GET `/api/report/cached/{role}`
**（推荐）** 获取缓存的 AI 报告，毫秒级响应。  
**(Recommended)** Return the cached report for a role — responds instantly.

支持的角色 / Supported roles: `teacher` | `principal` | `bureau` | `parent`

```
HTTP 200  → 缓存命中，返回完整报告 / Cache hit, full report returned
HTTP 202  → 仍在生成中，请稍后重试 / Still generating, retry later
```

> 报告在后端启动约 3 秒后开始为所有 4 个角色预生成，之后按 08:00 / 12:00 / 16:00 定时刷新。  
> Reports are pre-generated 3 seconds after startup for all 4 roles, then refreshed daily at 08:00 / 12:00 / 16:00.

### POST `/api/audio/detect-cough`
检测音频中的咳嗽（YAMNet + 自训练分类头）。  
Detect cough in uploaded audio (YAMNet + fine-tuned keras head).

```
Form field: file (audio/wav, audio/webm, audio/ogg, ...)

Response:
{ "probability": 0.874, "is_cough": true, "label": "cough", "confidence": "87.4%" }
```

### POST `/api/voiceprint/enroll`
注册说话人声纹。  
Enroll a speaker voiceprint.

```
Form fields: name (str), file (audio/*)
⚠️ 需要 ffmpeg / Requires ffmpeg
```

### POST `/api/voiceprint/verify`
识别说话人身份（余弦相似度 ≥ 0.72 视为匹配）。  
Verify speaker identity (cosine similarity ≥ 0.72 to match).

```
Form field: file (audio/*)
```

### GET `/api/health`
检查 Ollama 连通性与 RAG 状态。  
Check Ollama connectivity and RAG engine status.

---

## 支持角色 · Supported Roles

| 角色 Role | role 参数 | 视角 / Perspective |
|---|---|---|
| 班主任 Class Teacher | `teacher` | 班级层面，关注单班学生 |
| 校长 Principal | `principal` | 全校层面，关注多班趋势 |
| 教育局 Bureau | `bureau` | 区域层面，跨校统计决策 |
| 家长 Parent | `parent` | 家庭层面，关注自己孩子 |

---

## 离线模式 · Offline Mode

| 组件 / Component | 联网 Online | 断网 Offline |
|---|---|---|
| Gemma 4（Ollama） | ✅ | ✅ |
| RAG 知识库 | ✅ | ✅ |
| 天气（Open-Meteo） | ✅ | ✅ 模拟回退 / mock fallback |
| 医院负载数据 | ✅ | ✅ 模拟数据 / simulated |
| 前端 Firebase | ✅ | ✅ 演示模式 / demo mode |

---

## 常见问题 · Troubleshooting

**声纹录入报 HTTP 422 / Voiceprint enrollment fails with HTTP 422**  
→ ffmpeg 未安装。执行 `brew install ffmpeg`（macOS）后重启后端。  
→ ffmpeg is not installed. Run `brew install ffmpeg` (macOS) then restart the backend.

**"Cannot connect to Ollama"**  
→ 确保 Ollama 正在运行：`ollama run gemma4:e4b`  
→ Make sure Ollama is running: `ollama run gemma4:e4b`

**`[Errno 48] Address already in use`（端口 8000 被占用）**  
→ 旧进程仍在运行。执行：`lsof -ti :8000 | xargs kill -9`，或直接运行 `bash stop.sh`  
→ Old process still running. Run: `lsof -ti :8000 | xargs kill -9`, or simply run `bash stop.sh`

**"RAG engine not ready"**  
→ 检查 `backend/knowledge_base/*.md` 文件是否存在。  
→ Check that `knowledge_base/*.md` files exist in the `backend/` directory.

**报告生成很慢 / AI report is slow**  
→ 首次启动 Ollama 需加载模型到内存（约 30 秒），后续请求更快。  
→ 若使用缓存接口 `/api/report/cached/{role}`，报告在后端启动时即已预生成，点击即时返回。  
→ Ollama loads the model into memory on first use (~30s). Use `/api/report/cached/{role}` for instant results — reports are pre-generated at startup.

**嵌入模型下载失败 / Embedding model download fails**  
→ 手动下载 / Download manually:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

**AI 报告内容为英文或格式异常 / AI report shows English only or garbled format**  
→ 报告回退到了预设的双语兜底内容，说明 Ollama 调用超时或失败。检查 Ollama 是否正常运行。  
→ Report fell back to the built-in bilingual fallback, which means the Ollama call timed out or failed. Check that Ollama is running correctly.

---

## 项目结构 · Project Structure

```
Hackathon/
├── start.sh                          # 一键启动 / One-click start
├── stop.sh                           # 一键停止 / One-click stop
├── backend/
│   ├── main.py                       # FastAPI 应用入口 / App entry point
│   ├── audio_engine.py               # 咳嗽检测 + 声纹识别 / Cough & voiceprint
│   ├── requirements.txt
│   ├── knowledge_base/               # RAG 知识库 Markdown 文件
│   └── cough-detector/
│       └── best_cough_classifier.keras
└── 流感卫士-(fluguard-ai)/           # React + Vite 前端
    ├── src/
    │   ├── App.tsx                   # 主界面 / Main UI
    │   └── main.tsx
    └── package.json
```
