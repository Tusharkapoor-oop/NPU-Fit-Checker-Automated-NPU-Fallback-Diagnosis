@echo off
REM NPU Fit Checker - Offline Tracking Dashboard (no pip install needed)
cd /d "%~dp0"
python tracker.py --port 8080
