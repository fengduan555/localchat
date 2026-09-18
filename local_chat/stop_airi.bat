@echo off
chcp 65001 >nul
title Airi Local Chat - Stopper
echo ============================================================
echo   Stop Airi services (Chat / TTS / RVC)
echo ============================================================
echo.
wsl.exe -d Ubuntu -e bash /home/fengduan/kokoro-tts/local_chat/stop_all.sh
echo.
echo ------------------------------------------------------------
echo  Note: the RVC WebUI (if open) will be closed too.
echo ------------------------------------------------------------
timeout /t 8 >nul
