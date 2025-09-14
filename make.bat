@echo off
REM ==========================
REM Simple Makefile for Windows
REM ==========================

if "%1"=="test" (
    pytest -q
    exit /b %errorlevel%
)

if "%1"=="app" (
    streamlit run app\streamlit_app.py
    exit /b %errorlevel%
)

if "%1"=="train" (
    python scripts/train.py
    exit /b %errorlevel%
)

if "%1"=="eval" (
    python scripts/eval.py
    exit /b %errorlevel%
)

echo Usage: make [test|app|train|eval]
