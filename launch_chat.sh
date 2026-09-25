#!/usr/bin/env bash
# SMEC Support Bot Web UI One-Click Launcher

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "================================================="
echo "   SMEC Customer Support AI Chatbot Launcher   "
echo "================================================="

PYTHON_BIN="python3"

# Check if current Python environment has required modules (flask)
if ! "$PYTHON_BIN" -c "import flask" &>/dev/null; then
    if [ -x "/usr/bin/python3" ] && /usr/bin/python3 -c "import flask" &>/dev/null; then
        echo "[Notice] Active environment missing Flask. Switching to system Python (/usr/bin/python3)..."
        PYTHON_BIN="/usr/bin/python3"
    else
        echo "[Notice] Installing required dependencies..."
        "$PYTHON_BIN" -m pip install -r requirements.txt
    fi
fi

# Start Web App
echo "Launching web server using $PYTHON_BIN..."
"$PYTHON_BIN" web_app.py
