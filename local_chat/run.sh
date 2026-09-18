#!/bin/bash
# 启动本地语音聊天 (自动确保 TTS/RVC 服务在跑)
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[1/2] 确保 TTS + RVC 服务 ..."
bash "$HERE/../rvc_work/start_tts.sh" start
echo "[2/2] 启动聊天服务 http://localhost:8770"
exec /home/fengduan/kokoro-tts/venv/bin/python "$HERE/app.py" --port 8770
