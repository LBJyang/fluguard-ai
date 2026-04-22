#!/bin/bash
# ─── FluGuard AI Stop Script / 一键停止脚本 ──────────────────────────────────

echo ""
echo "🛑 FluGuard AI — Stopping... / 停止中..."
echo "────────────────────────────────────────"

# Stop backend / 关停后端 (port 8000)
if lsof -ti :8000 &>/dev/null; then
  lsof -ti :8000 | xargs kill -9 2>/dev/null
  echo "✅ Backend stopped. / 后端已停止 (port 8000)"
else
  echo "ℹ️  Backend not running. / 后端未在运行"
fi

# Stop frontend / 关停前端 (port 3000)
if lsof -ti :3000 &>/dev/null; then
  lsof -ti :3000 | xargs kill -9 2>/dev/null
  echo "✅ Frontend stopped. / 前端已停止 (port 3000)"
else
  echo "ℹ️  Frontend not running. / 前端未在运行"
fi

echo "────────────────────────────────────────"
echo "✅ All stopped. / 全部已停止"
echo ""
