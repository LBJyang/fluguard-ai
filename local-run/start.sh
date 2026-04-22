#!/bin/bash
# ─── FluGuard AI 一键启动脚本（本地版）────────────────────────────────────────
# 使用方法 / Usage:
#   bash start.sh
#
# 前提 / Prerequisites:
#   1. Ollama 已安装并已拉取模型：ollama pull gemma4:e4b
#   2. Python 虚拟环境已创建并安装依赖：
#      cd backend && python -m venv .venv && source .venv/bin/activate
#      pip install -r requirements.txt
#   3. 前端依赖已安装：cd frontend && npm install

# 脚本所在目录（支持从任意位置调用）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV="$BACKEND_DIR/.venv/bin/activate"

echo ""
echo "🛡  FluGuard AI — 启动中 / Starting..."
echo "────────────────────────────────────────"

# ── 1. 检查并关停已有的 8000 端口进程 ──────────────────────────────────────
if lsof -ti :8000 &>/dev/null; then
  echo "⚠️  端口 8000 已被占用，正在关停旧进程..."
  lsof -ti :8000 | xargs kill -9 2>/dev/null
  sleep 1
fi

# ── 2. 检查 Ollama 是否在运行 ───────────────────────────────────────────────
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "⚠️  Ollama 未运行，正在后台启动 gemma4:e4b..."
  ollama run gemma4:e4b &>/dev/null &
  sleep 5
else
  echo "✅ Ollama 已在运行"
fi

# ── 3. 启动后端（新终端窗口）──────────────────────────────────────────────
echo "🚀 启动后端 (FastAPI + Ollama + RAG)..."
osascript <<EOF
tell application "Terminal"
  do script "source $VENV && cd $BACKEND_DIR && python main.py"
  set custom title of front window to "FluGuard Backend"
end tell
EOF

# 等待后端就绪
echo "⏳ 等待后端启动..."
for i in {1..15}; do
  if curl -s http://localhost:8000/api/health &>/dev/null; then
    echo "✅ 后端已就绪 → http://localhost:8000"
    break
  fi
  sleep 1
done

# ── 4. 启动前端（新终端窗口）──────────────────────────────────────────────
echo "🚀 启动前端 (Vite + React)..."
osascript <<EOF
tell application "Terminal"
  do script "cd \"$FRONTEND_DIR\" && npm run dev"
  set custom title of front window to "FluGuard Frontend"
end tell
EOF

sleep 4

echo ""
echo "────────────────────────────────────────"
echo "✅ 全部启动完成！"
echo "   前端：http://localhost:5173"
echo "   后端：http://localhost:8000"
echo "   关停：bash stop.sh"
echo "────────────────────────────────────────"
echo ""

# 自动打开浏览器
sleep 2
open http://localhost:5173 2>/dev/null || open http://localhost:3000
