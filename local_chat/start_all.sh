#!/bin/bash
# 一键启动: TTS(8765) + 聊天服务(8770); RVC GPU 服务(8766) 由 TTS 首次请求自动拉起
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/app.log"
PY=/home/fengduan/kokoro-tts/venv/bin/python

health() { curl -s --max-time 3 "$1" -o /dev/null -w "%{http_code}" 2>/dev/null; }

echo "[1/3] TTS + RVC 服务 (8765/8766) ..."
bash "$HERE/../rvc_work/start_tts.sh" start

echo "[2/3] 聊天服务 (8770) ..."
if [ "$(health http://localhost:8770/api/config)" = "200" ]; then
  echo "      已在运行"
else
  setsid nohup "$PY" "$HERE/app.py" --port 8770 >> "$LOG" 2>&1 < /dev/null &
  for i in $(seq 1 25); do
    sleep 1
    [ "$(health http://localhost:8770/api/config)" = "200" ] && break
  done
fi

echo "[3/3] 后台预热 (首次需 30~60 秒, 不阻塞) ..."
(setsid nohup curl -s --max-time 180 -X POST http://localhost:8770/api/tts \
  -H "Content-Type: application/json" -d '{"text":"预热。"}' \
  -o /dev/null >/dev/null 2>&1 &)

GW=$(ip route | awk '/default/{print $3; exit}')
echo
echo "  状态总览:"
echo "    TTS   http://localhost:8765   -> $(health http://localhost:8765/health)"
echo "    Chat  http://localhost:8770   -> $(health http://localhost:8770/api/config)"
echo "    RVC   ws://$GW:8766           -> $(curl -s --max-time 4 http://$GW:8766/health 2>/dev/null | head -c 60)"
echo
echo "  浏览器打开: http://localhost:8770"
