@echo off
chcp 65001 >nul
title Airi Local Chat - Launcher
echo ============================================================
echo   Airi local voice chat   (LLM - Kokoro - RVC)
echo ============================================================
echo.
echo [1/2] Starting WSL services (TTS 8765 / Chat 8770 / RVC 8766)
echo.
wsl.exe -d Ubuntu -e bash /home/fengduan/kokoro-tts/local_chat/start_all.sh
echo.
echo [2/2] Opening browser ...
if /I "%~1"=="nobrowser" goto skipbrowser
start "" http://localhost:8770
:skipbrowser
echo.
echo ------------------------------------------------------------
echo  Chat page  : http://localhost:8770
echo  Stop       : run stop_airi.bat
echo  Note       : services stay in background; window can be closed
echo ------------------------------------------------------------
timeout /t 8 >nul
