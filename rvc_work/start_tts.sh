#!/bin/bash
# AIRI 语音服务启动/守护脚本 (WSL 侧)
# 用法: bash start_tts.sh start|stop|status
cd /home/fengduan/kokoro-tts || exit 1
LOG=/home/fengduan/kokoro-tts/rvc_work/tts_server.log
PORT=8765

health() {
  curl -s --max-time 3 "http://localhost:${PORT}/health" -o /dev/null -w "%{http_code}" 2>/dev/null
}

case "$1" in
  start)
    if [ "$(health)" = "200" ]; then echo "已在运行 (port $PORT)"; exit 0; fi
    echo "启动 tts_server (port $PORT) ..."
    nohup /home/fengduan/kokoro-tts/venv/bin/python \
      /home/fengduan/kokoro-tts/tts_server.py --port "$PORT" \
      >> "$LOG" 2>&1 &
    for i in $(seq 1 60); do
      sleep 2
      if [ "$(health)" = "200" ]; then echo "OK: http://localhost:${PORT}/v1 (就绪于 ${i}x2s)"; exit 0; fi
    done
    echo "启动超时, 看日志: $LOG"; exit 1
    ;;
  stop)
    P="tts_server.p""y"; pkill -f "$P" 2>/dev/null
    echo "已停止"
    ;;
  status)
    if [ "$(health)" = "200" ]; then echo "运行中"; else echo "未运行"; fi
    ;;
  *)
    echo "用法: $0 start|stop|status"; exit 1
    ;;
esac
