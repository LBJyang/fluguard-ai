#!/bin/bash
# ─── FluGuard AI 一键停止脚本 ────────────────────────────────────────────────

echo ""
echo "🛑 FluGuard AI — 停止中 / Stopping..."
echo "────────────────────────────────────────"

# 关停后端 (port 8000)
if lsof -ti :8000 &>/dev/null; then
  lsof -ti :8000 | xargs kill -9 2>/dev/null
  echo "✅ 后端已停止 (port 8000)"
else
  echo "ℹ️  后端未在运行"
fi

# 关停前端 Vite (port 5173 / 3000)
for PORT in 5173 3000 4173; do
  if lsof -ti :$PORT &>/dev/null; then
    lsof -ti :$PORT | xargs kill -9 2>/dev/null
    echo "✅ 前端已停止 (port $PORT)"
  fi
done

echo "────────────────────────────────────────"
echo "✅ 全部已停止"
echo ""
