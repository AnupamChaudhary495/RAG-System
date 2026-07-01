@echo off
title RAG Assistant
echo Starting RAG Assistant...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
echo.
echo RAG Assistant has stopped.
pause
