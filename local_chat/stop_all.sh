#!/bin/bash
# 停止全部: 聊天服务(8770) + TTS 服务(8765) + Windows RVC 服务(8766)
# 注意: 只杀监听 8766 的那个进程, 不再 taskkill 全部 python.exe
#       (否则会把 RVC WebUI 一起关掉)

P="local_chat/app.p""y";  pkill -f "$P" 2>/dev/null && echo "聊天服务已停" || echo "聊天服务未运行"
P2="tts_server.p""y";     pkill -f "$P2" 2>/dev/null && echo "TTS 服务已停" || echo "TTS 服务未运行"

# 按端口精确结束 Windows RVC 转换服务
PID=$(powershell.exe -NoProfile -Command \
  "(Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess" \
  2>/dev/null | tr -d '\r\n ' )
if [ -n "$PID" ] && [ "$PID" != "0" ]; then
  taskkill.exe /PID "$PID" /F >/dev/null 2>&1 && echo "RVC 服务已停 (PID $PID)" || echo "RVC 服务结束失败 (PID $PID)"
else
  echo "RVC 服务未运行"
fi

# 清理残留的看门狗循环 (如果有)
pgrep -f "start_tts.sh" | while read -r p; do kill "$p" 2>/dev/null; done
exit 0
