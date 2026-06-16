# token-looter (Zero-Config JSON-driven FastAPI Backend)

A high-performance, zero-configuration API proxy gateway for LLMs. It runs in a headless environment, using thread-safe in-memory caching and rotation logic.


## Key Architecture Features
* **Zero-Configuration Key Loading**: API keys are loaded directly from a plain-text configuration file (`keys.json`) at the project root directory. No database setups, settings tables, or encryption keys are required.
* **Stateless Fallback & Key Rotation**: Outbound requests sequentially rotate through all active keys for a given platform. Under error/cooldown events, failovers occur instantly in memory.
* **Headless Endpoint Shim**: Served purely as an API gateway for `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, and `/v1/responses`. All React frontend dashboard paths and administrative endpoints are deleted.
* **Analytics Logging**: SQLite is retained *exclusively* for logging analytics request histories (input/output tokens, latency, status).

## Setup & Run

### Prerequisites
- Python 3.10+
- `pip`

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Keys
Create a `keys.json` file in the root folder of the project containing your unified API gateway access key and individual provider keys:
```json
{
  "unified_api_key": "your_custom_gateway_access_token_here",
  "providers": {
    "google": ["AIzaSy..."],
    "groq": ["abc..."],
    "cerebras": [],
    "nvidia": [],
    "mistral": [],
    "openrouter": [],
    "github": [],
    "cohere": [],
    "cloudflare": [],
    "zhipu": [],
    "huggingface": [],
    "ollama": [],
    "kilo": [],
    "pollinations": [],
    "llm7": [],
    "opencode": [],
    "ovh": [],
    "custom": []
  }
}
```

### Run Server
```bash
# Start the server on port 3001
uvicorn app.main:app --host 0.0.0.0 --port 3001
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

- `DATABASE_PATH`: Custom path to the SQLite analytics database. Defaults to `data/freeapi.db`.
- `PROXY_URL`: Global outbound proxy URL (e.g. `http://proxy:8080`).
- `TOKENLOOTER_CONTEXT_HANDOFF`: Set to `on_model_switch` to enable context injection when a conversation switches models.
