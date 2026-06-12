"""
Health check and monitoring endpoints
Enterprise-grade observability & diagnostics
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from typing import Dict, Any
from datetime import datetime, timezone
import psutil
import platform
import socket
import os
import time

from backend.app.dependencies import (
    get_embedding_model,
    get_vector_store
)
from backend.utils.logger import setup_logger

router = APIRouter(
    prefix="/health",
    tags=["Monitoring"]
)

logger = setup_logger(__name__)

# =========================================================
# APP START TIME
# =========================================================

APP_START_TIME = time.time()

# =========================================================
# BASIC HEALTH CHECK
# =========================================================

@router.get("/")
async def health_check():
    """
    Lightweight health check
    Used for load balancers
    """

    return {
        "status": "healthy",
        "service": "INDIA RUNS Candidate Engine",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "uptime_seconds": round(
            time.time() - APP_START_TIME,
            2
        )
    }


# =========================================================
# DETAILED HEALTH CHECK
# =========================================================

@router.get("/detailed")
async def detailed_health(
    embedding_model=Depends(get_embedding_model),
    vector_store=Depends(get_vector_store)
):
    """
    Detailed service diagnostics
    """

    overall_status = "healthy"

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "service": {
            "name": "INDIA RUNS Candidate Engine",
            "environment": os.getenv(
                "ENVIRONMENT",
                "development"
            ),
            "version": os.getenv(
                "APP_VERSION",
                "1.0.0"
            ),
            "uptime_seconds": round(
                time.time() - APP_START_TIME,
                2
            ),
            "hostname": socket.gethostname()
        },
        "components": {},
        "system_metrics": {}
    }

    # -----------------------------------------------------
    # EMBEDDING MODEL CHECK
    # -----------------------------------------------------

    try:

        model_name = getattr(
            embedding_model,
            "model_name",
            "unknown"
        )

        health_status["components"][
            "embedding_model"
        ] = {
            "status": "healthy",
            "model_name": model_name
        }

    except Exception as e:

        overall_status = "degraded"

        health_status["components"][
            "embedding_model"
        ] = {
            "status": "unhealthy",
            "error": str(e)
        }

        logger.exception(
            "Embedding model health check failed"
        )

    # -----------------------------------------------------
    # VECTOR STORE CHECK
    # -----------------------------------------------------

    try:

        vector_count = (
            vector_store.get_size()
            if vector_store
            else 0
        )

        health_status["components"][
            "vector_store"
        ] = {
            "status": "healthy",
            "indexed_candidates": vector_count
        }

    except Exception as e:

        overall_status = "degraded"

        health_status["components"][
            "vector_store"
        ] = {
            "status": "unhealthy",
            "error": str(e)
        }

        logger.exception(
            "Vector store health check failed"
        )

    # -----------------------------------------------------
    # SYSTEM METRICS
    # -----------------------------------------------------

    try:

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=0.5)

        health_status["system_metrics"] = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(
                memory.available / (1024 ** 3),
                2
            ),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(
                disk.free / (1024 ** 3),
                2
            ),
            "process_count": len(
                psutil.pids()
            ),
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }

        # Auto-degrade on resource pressure
        if (
            cpu_percent > 90
            or memory.percent > 90
            or disk.percent > 95
        ):
            overall_status = "degraded"

    except Exception as e:

        overall_status = "degraded"

        health_status["system_metrics"] = {
            "status": "unavailable",
            "error": str(e)
        }

    health_status["status"] = overall_status

    return health_status


# =========================================================
# READINESS PROBE
# =========================================================

@router.get("/readiness")
async def readiness_check():
    """
    Kubernetes readiness probe
    """

    required_files = [
        "data/processed/cleaned_candidates.parquet",
        "vector_db/faiss.index"
    ]

    missing_files = [
        path
        for path in required_files
        if not os.path.exists(path)
    ]

    ready = len(missing_files) == 0

    response = {
        "ready": ready,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "missing_dependencies": missing_files
    }

    if not ready:

        raise HTTPException(
            status_code=503,
            detail=response
        )

    return response


# =========================================================
# LIVENESS PROBE
# =========================================================

@router.get("/liveness")
async def liveness_probe():
    """
    Kubernetes liveness probe
    """

    return {
        "alive": True,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


# =========================================================
# PROMETHEUS METRICS
# =========================================================

@router.get(
    "/metrics",
    response_class=PlainTextResponse
)
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint
    """

    process = psutil.Process(os.getpid())

    metrics = f"""
# HELP app_uptime_seconds Application uptime
# TYPE app_uptime_seconds counter
app_uptime_seconds {time.time() - APP_START_TIME}

# HELP app_cpu_percent CPU usage percent
# TYPE app_cpu_percent gauge
app_cpu_percent {psutil.cpu_percent(interval=0.1)}

# HELP app_memory_percent Memory usage percent
# TYPE app_memory_percent gauge
app_memory_percent {psutil.virtual_memory().percent}

# HELP process_memory_mb Process memory usage in MB
# TYPE process_memory_mb gauge
process_memory_mb {process.memory_info().rss / (1024 * 1024)}

# HELP process_threads Number of active threads
# TYPE process_threads gauge
process_threads {process.num_threads()}
"""

    return metrics.strip()


# =========================================================
# SYSTEM INFO
# =========================================================

@router.get("/system")
async def system_info():
    """
    Useful operational debugging information
    """

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "boot_time": datetime.fromtimestamp(
            psutil.boot_time(),
            tz=timezone.utc
        ).isoformat(),
        "environment": os.getenv(
            "ENVIRONMENT",
            "development"
        ),
        "app_version": os.getenv(
            "APP_VERSION",
            "1.0.0"
        )
    }


# =========================================================
# PERFORMANCE SNAPSHOT
# =========================================================

@router.get("/performance")
async def performance_snapshot():
    """
    Quick runtime performance diagnostics
    """

    process = psutil.Process(os.getpid())

    return {
        "cpu_percent": psutil.cpu_percent(
            interval=0.2
        ),
        "memory_percent": psutil.virtual_memory().percent,
        "process_memory_mb": round(
            process.memory_info().rss / (1024 * 1024),
            2
        ),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
        "uptime_seconds": round(
            time.time() - APP_START_TIME,
            2
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }