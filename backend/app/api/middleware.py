"""Request Logging and Privacy Middleware.

Specification Section 2, Section 7, Section 8, Rule R11:
- Generates a unique request_id per incoming HTTP request.
- Logs: request_id, method, path template/path, status code, duration in milliseconds.
- Strict Invariant: NEVER logs request bodies, response bodies, patient text,
  Authorization headers, or cookies.
"""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("clinexus.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing structured privacy-safe HTTP access logging."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        response: Response
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.error(
                "HTTP %s %s completed with 500 in %sms [request_id=%s]",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id

        # Determine path template if route exists
        path_template = request.url.path
        if "route" in request.scope and hasattr(request.scope["route"], "path"):
            path_template = request.scope["route"].path

        logger.info(
            "HTTP %s %s [%s] -> %s (%sms)",
            request.method,
            path_template,
            request_id,
            response.status_code,
            duration_ms,
        )

        return response
