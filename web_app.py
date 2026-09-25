"""
SMEC Customer Support Chatbot -- Web GUI Server
Serves a responsive glassmorphism web application for SMEC Support Chatbot.
"""

import json
import os
import sys
import webbrowser
from pathlib import Path
from threading import Timer

from flask import Flask, render_template, request, jsonify

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent))

from support_bot import (
    load_knowledge_base,
    build_system_prompt,
    get_working_groq_model,
    check_ollama_alive,
    call_ollama,
    FREE_GROQ_MODELS,
)

app = Flask(__name__)

KB = load_knowledge_base()
SYSTEM_PROMPT = build_system_prompt(KB)


def get_active_engine():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    force_ollama = os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    default_model = os.environ.get("GROQ_MODEL")

    ollama_ready = check_ollama_alive(ollama_model)

    if groq_api_key and not force_ollama:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            model_name = get_working_groq_model(client, default_model)
            return {
                "provider": "groq",
                "model": model_name,
                "client": client,
                "status": "Online (Groq Cloud)",
            }
        except Exception as e:
            pass

    if ollama_ready:
        return {
            "provider": "ollama",
            "model": ollama_model,
            "client": None,
            "status": "Local (Ollama)",
        }

    return {
        "provider": "none",
        "model": None,
        "client": None,
        "status": "No Active Engine",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/api/kb", methods=["GET"])
def get_kb_info():
    engine = get_active_engine()
    return jsonify({
        "company": KB.get("company", {}),
        "products": KB.get("products", []),
        "engine": {
            "provider": engine["provider"],
            "model": engine["model"],
            "status": engine["status"],
            "available_groq_models": FREE_GROQ_MODELS,
        }
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    history = data.get("messages", [])
    user_message = data.get("message", "").strip()
    requested_model = data.get("model", "").strip()

    if not user_message and not history:
        return jsonify({"error": "Empty message"}), 400

    # Build messages array with system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            full_messages.append({"role": msg["role"], "content": msg["content"]})

    if user_message:
        full_messages.append({"role": "user", "content": user_message})

    engine = get_active_engine()
    provider = engine["provider"]
    model_name = requested_model or engine["model"]

    # Handle explicit provider switching via model prefix
    if requested_model.startswith("ollama:"):
        provider = "ollama"
        model_name = requested_model.split(":", 1)[1]
    elif requested_model and provider == "groq":
        model_name = requested_model

    if provider == "none":
        return jsonify({
            "error": "No LLM Engine ready. Please set GROQ_API_KEY environment variable or start local Ollama server."
        }), 503

    reply = None
    used_model = model_name

    if provider == "groq":
        client = engine["client"]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=full_messages,
                temperature=0.3,
                max_tokens=600,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            # Fallback across candidate models
            for alt in FREE_GROQ_MODELS:
                if alt == model_name:
                    continue
                try:
                    response = client.chat.completions.create(
                        model=alt,
                        messages=full_messages,
                        temperature=0.3,
                        max_tokens=600,
                    )
                    reply = response.choices[0].message.content
                    used_model = alt
                    break
                except Exception:
                    continue

            # Fallback to Ollama if Groq failed
            ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
            if not reply and check_ollama_alive(ollama_model):
                try:
                    reply = call_ollama(full_messages, ollama_model)
                    provider = "ollama"
                    used_model = ollama_model
                except Exception:
                    pass

    elif provider == "ollama":
        try:
            reply = call_ollama(full_messages, model_name)
        except Exception as e:
            # Fallback to Groq if Ollama failed
            groq_key = os.environ.get("GROQ_API_KEY")
            if groq_key:
                try:
                    from groq import Groq
                    g_client = Groq(api_key=groq_key)
                    g_model = get_working_groq_model(g_client)
                    response = g_client.chat.completions.create(
                        model=g_model,
                        messages=full_messages,
                        temperature=0.3,
                        max_tokens=600,
                    )
                    reply = response.choices[0].message.content
                    provider = "groq"
                    used_model = g_model
                except Exception:
                    pass

    if reply:
        return jsonify({
            "reply": reply,
            "provider": provider,
            "model": used_model
        })
    else:
        return jsonify({
            "error": "Failed to generate response. Please check your network connection or API key."
        }), 500


def open_browser():
    webbrowser.open_new("http://localhost:5000/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Open browser automatically if running interactively
    if "--no-browser" not in sys.argv:
        Timer(1.2, open_browser).start()

    print(f"Starting SMEC Customer Support Web UI on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
