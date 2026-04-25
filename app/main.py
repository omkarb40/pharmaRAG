"""
PharmaRAG — FastAPI application entry point.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.routers import ask
from app.routers import monitoring

app = FastAPI(
    title="PharmaRAG",
    description="RAG Reliability & Governance Framework for Drug Safety QA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router, prefix="/api", tags=["QA"])
app.include_router(monitoring.router, prefix="/api", tags=["Monitoring"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "pharma-rag"}


@app.get("/", response_class=HTMLResponse)
def chat_page():
    """Serve the chat interface."""
    html_path = Path(__file__).parent / "chat.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)