"""
SMEC customer support chatbot -- small knowledge base, LLM reasoning via Groq (free tier) or local Ollama.

Setup for Groq (Free Tier):
    pip install groq --break-system-packages
    export GROQ_API_KEY="your_free_key_from_console.groq.com"

Available Free Models on Groq:
    - llama-3.1-8b-instant           (Recommended - fast, high quality)
    - gemma2-9b-it                  (Google Gemma 2 9B)
    - deepseek-r1-distill-llama-70b (DeepSeek R1 Distill 70B)
    - qwen-2.5-32b                  (Qwen 2.5 32B)
    - mixtral-8x7b-32768            (Mistral 8x7B)

Run:
    python3 support_bot.py
    # Or to use a specific model:
    GROQ_MODEL=gemma2-9b-it python3 support_bot.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

KB_PATH = Path(__file__).parent / "knowledge_base.json"

GROK_MODELS = [
    "grok-2-1212",
    "grok-beta",
    "grok-2-vision-1212",
]

FREE_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

DEFAULT_MODEL = os.environ.get("GROQ_MODEL") or os.environ.get("GROK_MODEL")


def call_grok(messages: list, model_name: str = "grok-2-1212", api_key: str = None) -> str:
    """Call xAI Grok API endpoint with error handling."""
    key = api_key or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        raise ValueError("GROK_API_KEY or XAI_API_KEY is missing from environment or .env file.")

    import urllib.request

    url = "https://api.xai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": "SMECBot",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 600,
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_working_groq_model(client, preferred: str = None) -> str:
    """Query Groq API for currently active chat models and return the best working model."""
    NON_CHAT_KEYWORDS = ["guard", "whisper", "vision", "embed", "eval", "safeguard", "prompt-guard", "orpheus", "canopylabs"]
    try:
        models_data = client.models.list().data
        available_map = {m.id.lower(): m.id for m in models_data}
        
        if preferred and preferred.lower() in available_map:
            return available_map[preferred.lower()]
            
        for cand in FREE_GROQ_MODELS:
            if cand.lower() in available_map:
                return available_map[cand.lower()]
                
        # Fallback to standard active chat models
        for m in models_data:
            mid = m.id.lower()
            if not any(kw in mid for kw in NON_CHAT_KEYWORDS):
                return m.id
    except Exception:
        pass

    return preferred or "qwen/qwen3.8-27b"


def check_ollama_alive(model_name: str = "llama3.2") -> bool:
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", headers={"User-Agent": "SMECBot"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama(messages: list, model_name: str = "llama3.2") -> str:
    import urllib.request
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps({"model": model_name, "messages": messages, "stream": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res["message"]["content"]


def build_system_prompt(kb: dict) -> str:
    kb_json = json.dumps(kb, indent=2)

    return f"""You are the customer support assistant for {kb['company']['name']}, a marine and industrial automation company.

You are talking to real customers -- likely marine engineers, ship operators, procurement staff, or technicians -- who need accurate information about SMEC's products and services. Getting a technical spec wrong could lead to someone ordering the wrong part or misconfiguring equipment on a vessel. Be precise and honest about what you do and don't know.

## KNOWLEDGE BASE
This is the ONLY information you have about SMEC's products and company. Do not use outside knowledge about marine automation, generic industry specs, or competitor products to fill gaps.

{kb_json}

## HOW TO REASON WITH CUSTOMERS

1. **Ground every claim in the knowledge base above.** If a customer asks about a spec, feature, or product that isn't in the knowledge base, say so plainly -- do not estimate, infer, or use general industry knowledge to fill the gap. A wrong guess is worse than "I don't have that on file."

2. **Distinguish confirmed specs from general descriptions.** Some products above (like the Battery Charger) have full spec tables you can quote exactly. Others (like the Salinity Monitoring System) only have a feature list -- if a customer asks for a number that isn't listed (e.g. "what's the measurement range?"), say that detail isn't in what you have, rather than approximating.

3. **Quote numbers exactly.** Never round, convert units, or approximate a spec from the knowledge base. If asked to convert units, do the math but show your source figure first (e.g. "According to the datasheet, output is 24V DC... which is [conversion] if you need that in different units").

4. **Ask a clarifying question when the request is ambiguous** -- e.g. if a customer asks "what's the input voltage" without naming a product, ask which product they mean rather than guessing.

5. **Know when to escalate.** Hand off to a human (give the contact info from the knowledge base: {kb['company']['contact']['email_general']} or {kb['company']['contact']['phone']}) when:
   - The customer needs a quote, pricing, or order placement
   - The question requires a spec/detail not present in the knowledge base
   - The customer describes a fault, malfunction, or safety-critical issue with installed equipment
   - The customer is frustrated or the conversation isn't resolving their issue after a couple of exchanges

6. **Tone**: professional, concise, technically fluent -- match the register of someone who understands marine/industrial terminology. Don't oversell or use marketing fluff; customers asking support questions want facts, not a pitch.

7. **Stay in scope.** You represent SMEC's product/support function only -- don't offer opinions on competitors, don't speculate on delivery timelines or pricing (redirect to sales contact), and don't make commitments on SMEC's behalf (installation dates, warranty claims, etc.) -- escalate those instead.
"""


def chat_loop():
    kb = load_knowledge_base()
    system_prompt = build_system_prompt(kb)
    messages = [{"role": "system", "content": system_prompt}]

    groq_api_key = os.environ.get("GROQ_API_KEY")
    force_ollama = os.environ.get("USE_OLLAMA", "").lower() in ("1", "true", "yes")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    groq_client = None
    groq_model = None
    ollama_ready = check_ollama_alive(ollama_model)

    if groq_api_key and not force_ollama:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_api_key)
            groq_model = get_working_groq_model(groq_client, DEFAULT_MODEL)
        except Exception as e:
            print(f"Warning: Could not initialize Groq client: {e}")

    active_provider = None
    if groq_client and groq_model:
        active_provider = "groq"
        print(f"SMEC Support Bot ready [Engine: Groq Cloud ({groq_model})]. ({len(kb['products'])} products loaded). Type 'exit' to quit.\n")
    elif ollama_ready:
        active_provider = "ollama"
        print(f"SMEC Support Bot ready [Engine: Local Ollama ({ollama_model})]. ({len(kb['products'])} products loaded). Type 'exit' to quit.\n")
    else:
        print("ERROR: Neither Groq Cloud nor Local Ollama is ready.")
        print("\nTo set up Groq Cloud:")
        print("  export GROQ_API_KEY=\"your_groq_api_key_from_console.groq.com\"")
        print("  python3 support_bot.py")
        print("\nTo set up Local Ollama:")
        print("  1. Run 'ollama serve' in a separate terminal")
        print("  2. python3 support_bot.py")
        sys.exit(1)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        reply = None
        # Try active provider
        if active_provider == "groq":
            try:
                response = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                print(f"[Groq model '{groq_model}' error: {e}]")
                # Try fallback Groq models dynamically
                for alt in FREE_GROQ_MODELS:
                    if alt == groq_model:
                        continue
                    try:
                        print(f"Retrying with alternative Groq model '{alt}'...")
                        response = groq_client.chat.completions.create(
                            model=alt,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=600,
                        )
                        reply = response.choices[0].message.content
                        groq_model = alt
                        break
                    except Exception:
                        continue

                # If Groq models fail but Ollama is available, fallback to Ollama
                if not reply and check_ollama_alive(ollama_model):
                    print("Falling back to local Ollama...")
                    try:
                        reply = call_ollama(messages, ollama_model)
                        active_provider = "ollama"
                    except Exception as err:
                        print(f"[Ollama error: {err}]")

        elif active_provider == "ollama":
            try:
                reply = call_ollama(messages, ollama_model)
            except Exception as e:
                print(f"[Ollama error: {e}]")
                if groq_api_key:
                    print("Falling back to Groq Cloud...")
                    try:
                        from groq import Groq
                        groq_client = Groq(api_key=groq_api_key)
                        groq_model = get_working_groq_model(groq_client, DEFAULT_MODEL)
                        response = groq_client.chat.completions.create(
                            model=groq_model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=600,
                        )
                        reply = response.choices[0].message.content
                        active_provider = "groq"
                    except Exception as g_err:
                        print(f"[Groq fallback error: {g_err}]")

        if reply:
            print(f"\nBot: {reply}\n")
            messages.append({"role": "assistant", "content": reply})
        else:
            print("\n[Error: Unable to generate response from either Groq or local Ollama. Please check your setup.]\n")


if __name__ == "__main__":
    chat_loop()
