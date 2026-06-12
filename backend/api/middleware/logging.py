"""
Production-grade Request/Response Logging Middleware
"""

import time
import uuid
import traceback
from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs:
    - Request metadata
    - Response status
    - Processing duration
    - Errors/exceptions
    - Request IDs for tracing
    """

    async def dispatch(self, request: Request, call_next):

        # Skip noisy endpoints
        SKIP_PATHS = ["/health", "/docs", "/openapi.json"]

        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        start_time = time.time()

        # Unique request tracking ID
        request_id = str(uuid.uuid4())

        # Read request body safely
        try:
            body = await request.body()
            body_content = body[:2000].decode("utf-8", errors="ignore") if body else None
        except Exception:
            body_content = "Unable to decode request body"

        # Request metadata
        request_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "body": body_content,
        }

        logger.info(
            f"[REQUEST] {request.method} {request.url.path}",
            extra=request_data
        )

        try:
            # Process request
            response: Response = await call_next(request)

            # Processing duration
            duration_ms = round((time.time() - start_time) * 1000, 2)

            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }

            logger.info(
                f"[RESPONSE] {request.method} {request.url.path} "
                f"- {response.status_code} ({duration_ms}ms)",
                extra=response_data
            )

            # Add useful tracing headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms}ms"

            return response

        except Exception as e:

            duration_ms = round((time.time() - start_time) * 1000, 2)

            error_data = {
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration_ms": duration_ms,
            }

            logger.error(
                f"[ERROR] {request.method} {request.url.path} "
                f"failed after {duration_ms}ms - {str(e)}",
                extra=error_data
            )

            raise
