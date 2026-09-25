# SMEC Automation AI Customer Support Chatbot

An AI-powered Customer Support Chatbot system designed for **SMEC Automation Pvt Ltd** (Marine & Industrial Automation). The chatbot features a responsive Glassmorphism Web Interface, dynamic model selection (Groq Cloud LLMs & Local Ollama), grounding in a specialized knowledge base, and automated failovers.

---

## 🌟 Key Features

- **🎨 Glassmorphism Web Interface**: Dark mode UI with real-time model status indicators, dynamic sidebar card synchronization, copy-to-clipboard actions, quick suggested queries, and auto-scroll chat history.
- **⚡ Dual Engine Support**:
  - **Groq Cloud API**: Ultra-fast inference with free-tier open models (`llama-3.3-70b-versatile`, `llama-3.1-80b-instant`, `mixtral-8x7b-32768`, etc.).
  - **Local Ollama**: Privacy-first local LLM execution (`llama3.2`, `llama3.1`, `mistral`, etc.) without requiring external API keys.
- **🔄 Dynamic Model Switcher**: Change inference models on-the-fly directly from the web UI header dropdown.
- **📚 Grounded Knowledge Base**: Powered by a structured JSON (`knowledge_base.json`) detailing SMEC Automation services, marine products, technical specs, branches, contact details, and escalation procedures.
- **🛡️ Guardrails & Escalation**: Built-in instructions prevent hallucination of non-existent specifications and guide customers to human support for missing details.

---

## 📁 Repository Structure

```
smec_bot/
├── knowledge_base.json   # SMEC Automation company details, products & specs
├── support_bot.py        # Core LLM logic, Groq & Ollama API integrations, prompt builder
├── web_app.py            # Flask web server & REST API endpoints
├── launch_chat.sh        # One-click executable bash launcher
├── templates/
│   └── index.html        # Glassmorphism HTML/JS frontend
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the required Python packages:

```bash
pip install flask requests python-dotenv groq
```

### 2. Environment Setup (Optional for Groq Cloud)

Set your Groq API key if using Groq Cloud inference:

```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

*Note: If `GROQ_API_KEY` is not provided, the application will automatically fall back to local Ollama if running.*

If using **Ollama** locally, ensure Ollama is installed and serving models:

```bash
ollama run llama3.2
```

### 3. Launch the Application

Run the convenient one-click bash script:

```bash
chmod +x launch_chat.sh
./launch_chat.sh
```

Or start the Flask server directly:

```bash
python3 web_app.py
```

The web application will automatically open in your default browser at `http://127.0.0.1:5000`.

---

## ⚙️ Configuration & Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Groq Cloud API key for cloud LLM inference | None |
| `GROQ_MODEL` | Preferred default Groq model | Auto-detected |
| `USE_OLLAMA` | Set to `true` to prioritize local Ollama engine | `false` |
| `OLLAMA_MODEL` | Preferred default Ollama model name | `llama3.2` |
| `OLLAMA_HOST` | Ollama service base URL | `http://localhost:11434` |

---

## 🛠️ API Endpoints

- `GET /` — Serves the main web dashboard.
- `GET /api/info` — Returns current engine status, active provider, model list, and company metadata.
- `POST /api/chat` — Accepts JSON `{ "message": "...", "model": "..." }` and returns the AI support response.

---

## 👨‍💻 Author

Developed for **SMEC Automation Pvt Ltd**.
