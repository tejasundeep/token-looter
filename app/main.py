import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load env file
env_path = os.environ.get("FREEAPI_ENV_PATH")
if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    default_env = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=default_env)

from app.database import init_db
from app.lib.proxy import close_all_clients

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database for tracking logs
    db_path = os.environ.get("DATABASE_PATH")
    init_db(db_path)
    yield
    await close_all_clients()


app = FastAPI(lifespan=lifespan, title="token-looter")

# Allow all CORS for developer flexibility (IDE integrations)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": {"message": str(exc.errors()), "type": "invalid_request_error"}}
    )

# Include unified API router
from app.v1_endpoints import v1_router
app.include_router(v1_router)

print("[Server] token-looter started in headless mode (Zero-Config JSON-driven)")
