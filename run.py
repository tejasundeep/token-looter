"""
TokenLooter — local development server
Run with:  python run.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env before importing uvicorn so the encryption key is in the environment
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

import uvicorn  # noqa: E402 – must come after dotenv load

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")

    print(f"\n  TokenLooter API  ->  http://{host}:{port}")
    print(f"  Dashboard        ->  http://{host}:{port}/dashboard")
    print(f"  OpenAI base URL  ->  http://{host}:{port}/v1\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,             # hot-reload on file changes
        reload_dirs=["app"],
        log_level="info",
    )
