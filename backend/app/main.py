"""Application Factory and Entry Point.

Specification Section 2, Section 7, Section 8, Rule R11:
- create_app() factory.
- Lifespan startup: validates settings, runs reconcile_interrupted_executions(), creates first admin.
- CORS configured strictly from CORS_ALLOWED_ORIGINS (never wildcard).
- Request logging middleware (privacy-preserving, no bodies or tokens).
- Exception handlers formatting errors strictly as: {error: {code, message, details}}.
- Strips 'input' parameter from validation errors (no patient free text echoing).
- OpenAPI specification hosted at /api/v1/openapi.json.
- Prefix: /api/v1.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.errors import (
    AppAPIException,
    sanitize_validation_errors,
)
from backend.app.api.middleware import RequestLoggingMiddleware
from backend.app.api.router import api_router
from backend.app.core.config import load_settings, settings
from backend.app.orchestration.states import InvalidTransition
from backend.app.persistence.database import SessionLocal
from backend.app.persistence.execution_repository import reconcile_interrupted_executions
from backend.app.security.auth_service import ensure_first_admin
from backend.app.security.roles import UnauthorizedActionError

logger = logging.getLogger("clinexus.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle management."""
    # 1. Validate configuration settings
    try:
        active_settings = load_settings()
        logger.info("Application settings validated successfully. Environment: %s", active_settings.ENVIRONMENT)
    except Exception as exc:
        logger.critical("Configuration validation failed at startup: %s", exc)
        raise

    # 2. Reconcile interrupted executions from previous runs (SPEC 2, DB-3)
    try:
        reconciled = reconcile_interrupted_executions()
        if reconciled > 0:
            logger.warning("Reconciled %d interrupted workflow executions to FAILED status.", reconciled)
    except Exception as exc:
        logger.error("Failed to reconcile interrupted executions: %s", exc)

    # 3. Create initial administrator account only if no user exists (SPEC 7)
    try:
        with SessionLocal() as session:
            admin_created = ensure_first_admin(session)
            if admin_created:
                logger.info("First admin account provisioned at startup.")
    except Exception as exc:
        logger.error("Admin account verification failed: %s", exc)

    yield

    logger.info("Application shutdown completed.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Clinexus API",
        description="Evidence-Grounded Clinical Decision Support Platform REST API",
        version="1.0.0",
        openapi_url="/api/v1/openapi.json",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # 1. CORS Configuration (Never wildcard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Privacy-Safe Request Logging Middleware
    app.add_middleware(RequestLoggingMiddleware)

    # 3. Exception Handlers Formatting Strictly as {error: {code, message, details}}

    @app.exception_handler(AppAPIException)
    async def app_api_exception_handler(request: Request, exc: AppAPIException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        sanitized_details = sanitize_validation_errors(exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Validation error in request payload or parameters.",
                    "details": sanitized_details,
                }
            },
        )

    @app.exception_handler(InvalidTransition)
    async def invalid_transition_handler(request: Request, exc: InvalidTransition) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "INVALID_STATE",
                    "message": str(exc),
                    "details": None,
                }
            },
        )

    @app.exception_handler(UnauthorizedActionError)
    async def unauthorized_action_handler(request: Request, exc: UnauthorizedActionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "code": "FORBIDDEN",
                    "message": str(exc),
                    "details": None,
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code_map = {
            401: "UNAUTHENTICATED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            409: "CONFLICT",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMITED",
            500: "INTERNAL_ERROR",
        }
        err_code = code_map.get(exc.status_code, "HTTP_ERROR")
        msg = exc.detail if isinstance(exc.detail, str) else "HTTP request error."
        details = exc.detail if isinstance(exc.detail, (dict, list)) else None

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": err_code,
                    "message": msg,
                    "details": details,
                }
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal server error occurred.",
                    "details": None,
                }
            },
        )

    # 4. Mount API Routes with /api/v1 Prefix
    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()
