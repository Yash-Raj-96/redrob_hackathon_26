"""
FastAPI Main Application Entry Point
INDIA RUNS - Candidate Discovery Engine (SAFE VERSION)
"""

from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.middleware.logging import LoggingMiddleware
from backend.api.routes import analytics, explain, health, ranking, search
from backend.app.config import settings
from backend.app.dependencies import (
    get_candidates_data,
    get_embedding_model,
    get_ranker,
    get_retriever,
    get_vector_store
)
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


# =========================================================
# LIFESPAN (SAFE - NEVER CRASH APP)
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle (SAFE MODE)
    """

    logger.info("Starting INDIA RUNS Candidate Discovery Engine...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # =========================
    # SAFE STARTUP (NO CRASH)
    # =========================

    # 1. Embedding Model
    try:
        await get_embedding_model()
        logger.info("Embedding model initialized")
    except Exception as e:
        logger.error(f"Embedding model failed: {e}")

    # 2. Vector Store (optional)
    try:
        await get_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Vector store failed: {e}")

    # 3. Retriever (MOST IMPORTANT - MUST NOT CRASH APP)
    try:
        await get_retriever()
        logger.info("Retriever initialized")
    except Exception as e:
        logger.error(f"Retriever failed (degraded mode active): {e}")

    # 4. Ranker
    try:
        await get_ranker()
        logger.info("Ranker initialized")
    except Exception as e:
        logger.error(f"Ranker failed: {e}")

    # 5. Dataset (SAFE)
    try:
        df = await get_candidates_data()
        logger.info(f"Loaded {len(df)} candidates")
    except Exception as e:
        logger.error(f"Dataset load failed: {e}")

    logger.info("Application startup completed (SAFE MODE ACTIVE)")

    yield

    # =========================
    # SHUTDOWN
    # =========================

    logger.info("Shutting down application...")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="INDIA RUNS Candidate Discovery Engine",
    description="AI-powered candidate search, ranking, and analytics engine",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)


# =========================================================
# MIDDLEWARE
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)


# =========================================================
# ROUTES
# =========================================================

app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(ranking.router, prefix="/api/ranking", tags=["Ranking"])
app.include_router(explain.router, prefix="/api/explain", tags=["Explainability"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])


# =========================================================
# ROOT
# =========================================================

@app.get("/", tags=["Root"])
async def root():
    return {
        "application": settings.APP_NAME,
        "version": "2.0.0",
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/api/docs",
        "health": "/api/health",
        "mode": "safe_degraded_mode_enabled"
    }


# =========================================================
# PING
# =========================================================

@app.get("/ping", tags=["Health"])
async def ping():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }


# =========================================================
# GLOBAL EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.exception(f"Unhandled exception: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "Something went wrong",
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =========================================================
# VALUE ERROR HANDLER
# =========================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):

    logger.warning(f"Validation error: {str(exc)}")

    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Validation Error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )