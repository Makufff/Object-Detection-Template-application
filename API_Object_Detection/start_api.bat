@echo off
start cmd /k ngrok http 8000
start cmd /k python app.py