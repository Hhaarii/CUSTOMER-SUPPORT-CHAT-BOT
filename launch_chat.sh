#!/usr/bin/env bash
# SMEC Support Bot Web UI One-Click Launcher

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "================================================="
echo "   SMEC Customer Support AI Chatbot Launcher   "
echo "================================================="

# Start Web App
python3 web_app.py
