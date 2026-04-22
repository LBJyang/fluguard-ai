#!/bin/bash
# ─── FluGuard AI Quick-Start Script ──────────────────────────────────────────
# Auto-installs dependencies on first run. No manual path config needed.
# 一键启动脚本，首次运行自动安装依赖，无需手动配置路径。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
# Set FLUGUARD_VENV to use an existing virtualenv and skip installation.
# 可设置 FLUGUARD_VENV 指定已有虚拟环境以跳过安装，例如：
# FLUGUARD_VENV=/Users/you/ai-env sh start.sh
VENV="${FLUGUARD_VENV:+$FLUGUARD_VENV/bin/activate}"
VENV="${VENV:-$BACKEND_DIR/venv/bin/activate}"

echo ""
echo "🛡  FluGuard AI — Starting... / 启动中..."
echo "────────────────────────────────────────"

# ── 1. Check Python 3 / 检查 Python 3 ────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Please install Python 3.10+ first."
  echo "   未找到 python3，请先安装 Python 3.10+"
  exit 1
fi

# ── 2. Check Node.js / 检查 Node.js ──────────────────────────────────────────
if ! command -v node &>/dev/null; then
  echo "❌ node not found. Please install Node.js 18+ first."
  echo "   未找到 node，请先安装 Node.js 18+"
  exit 1
fi

# ── 3. Check Ollama / 检查 Ollama ─────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "❌ ollama not found. Please install from: https://ollama.com/download"
  echo "   未找到 ollama，请先安装：https://ollama.com/download"
  exit 1
fi

# ── 4. First-run: create Python venv and install dependencies
#       首次运行：创建 Python 虚拟环境并安装依赖 ─────────────────────────────
INSTALLED_MARKER="$BACKEND_DIR/venv/.installed"
if [ -n "$FLUGUARD_VENV" ]; then
  echo "✅ Using existing virtualenv: $FLUGUARD_VENV"
  echo "   使用指定虚拟环境：$FLUGUARD_VENV"
  source "$VENV"
elif [ ! -f "$INSTALLED_MARKER" ]; then
  echo "🔧 First run: creating Python virtualenv... / 首次运行：创建 Python 虚拟环境..."
  rm -rf "$BACKEND_DIR/venv"
  python3 -m venv "$BACKEND_DIR/venv"
  echo "📦 Installing backend dependencies (first run ~5-10 min, downloading models)..."
  echo "   安装后端依赖（首次约需 5-10 分钟，需下载模型文件）..."
  source "$VENV"
  pip install --quiet --upgrade pip
  pip install --quiet -r "$BACKEND_DIR/requirements.txt"
  touch "$INSTALLED_MARKER"
  echo "✅ Backend dependencies installed. / 后端依赖安装完成"
else
  source "$VENV"
  echo "✅ Backend dependencies ready (skipping install). / 后端依赖已就绪（跳过安装）"
fi

# ── 5. First-run: npm install / 首次运行：安装前端 npm 依赖 ──────────────────
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "📦 First run: installing frontend dependencies (npm install)..."
  echo "   首次运行：安装前端依赖..."
  cd "$FRONTEND_DIR" && npm install --silent
  echo "✅ Frontend dependencies installed. / 前端依赖安装完成"
fi

# ── 6. Free up ports if occupied / 关停已有的端口占用 ────────────────────────
for PORT in 8000 3000; do
  if lsof -ti :$PORT &>/dev/null; then
    echo "⚠️  Port $PORT in use, stopping old process... / 端口 $PORT 已被占用，正在关停旧进程..."
    lsof -ti :$PORT | xargs kill -9 2>/dev/null
    sleep 1
  fi
done

# ── 7. Start Ollama and pull model / 启动 Ollama + 拉取模型 ──────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "⚠️  Ollama not running, starting in background... / Ollama 未运行，正在后台启动..."
  ollama serve &>/dev/null &
  sleep 3
fi

if ! ollama list 2>/dev/null | grep -q "gemma4:e4b"; then
  echo "📥 First run: pulling model gemma4:e4b (~2.5 GB, please wait)..."
  echo "   首次运行：拉取模型 gemma4:e4b（约 2.5 GB，请耐心等待）..."
  ollama pull gemma4:e4b
fi
echo "✅ Ollama ready, model gemma4:e4b loaded. / Ollama 就绪，模型 gemma4:e4b 已加载"

# ── 8. Start backend in new terminal / 启动后端（新终端窗口）─────────────────
echo "🚀 Starting backend (FastAPI + Ollama + RAG)... / 启动后端..."
osascript <<EOF
tell application "Terminal"
  do script "source \"$VENV\" && cd \"$BACKEND_DIR\" && python main.py"
  set custom title of front window to "FluGuard Backend"
end tell
EOF

# Wait for backend / 等待后端就绪
echo "⏳ Waiting for backend to be ready... / 等待后端启动..."
for i in {1..20}; do
  if curl -s http://localhost:8000/api/health &>/dev/null; then
    echo "✅ Backend ready → http://localhost:8000 / 后端已就绪"
    break
  fi
  sleep 1
done

# ── 9. Start frontend in new terminal / 启动前端（新终端窗口）───────────────
echo "🚀 Starting frontend (Vite + React)... / 启动前端..."
osascript <<EOF
tell application "Terminal"
  do script "cd \"$FRONTEND_DIR\" && npm run dev"
  set custom title of front window to "FluGuard Frontend"
end tell
EOF

sleep 4

echo ""
echo "────────────────────────────────────────"
echo "✅ All services started! / 全部启动完成！"
echo "   Frontend / 前端：http://localhost:3000"
echo "   Backend  / 后端：http://localhost:8000"
echo "   Stop     / 关停：bash stop.sh"
echo "────────────────────────────────────────"
echo ""

sleep 2
open http://localhost:3000
