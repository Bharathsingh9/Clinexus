"""Health and Dependency Check Endpoints.

Specification Section 7, Section 13 (API-1):
- GET /health: Liveness probe.
- GET /health/dependencies: Readiness and dependency diagnostics.
  Checks: PostgreSQL, Qdrant (main collection and point count), LLM configuration, NCBI reachability.
  Invariants: Never exposes secrets, keys, or passwords. Reports down dependencies as DOWN
  without throwing unhandled crashes or stack traces.
"""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends
import httpx
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.api.deps import get_db
from backend.app.core.config import settings
from backend.app.retrieval.qdrant_client import format_collection_name

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Application Liveness Probe")
def get_liveness() -> Dict[str, str]:
    """Basic liveness probe returning OK if HTTP application loop is running."""
    return {"status": "ok"}


@router.get("/health/dependencies", summary="System Dependencies Readiness Probe")
def get_dependencies(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Inspect status of core infrastructure dependencies without printing secrets."""
    overall_ok = True
    dep_status: Dict[str, Any] = {}

    # 1. PostgreSQL Persistence Check
    try:
        db.execute(text("SELECT 1")).scalar()
        dep_status["postgres"] = "UP"
    except Exception as exc:
        logger.warning("Health check PostgreSQL failure: %s", exc)
        dep_status["postgres"] = "DOWN"
        overall_ok = False

    # 2. Qdrant Vector Storage Check
    try:
        coll_name = format_collection_name(
            settings.EMBEDDING_MODEL,
            settings.QDRANT_COLLECTION_VERSION,
        )
        qclient = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=3.0)
        qcount = qclient.count(collection_name=coll_name).count
        dep_status["qdrant"] = {
            "status": "UP",
            "collection": coll_name,
            "point_count": qcount,
        }
    except Exception as exc:
        logger.warning("Health check Qdrant failure: %s", exc)
        dep_status["qdrant"] = {
            "status": "DOWN",
        }
        overall_ok = False

    # 3. LLM Configuration Check (No secrets exposed)
    dep_status["llm"] = {
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL,
        "key_present": bool(settings.LLM_API_KEY),
    }

    # 4. NCBI E-utilities Reachability Check
    if settings.EXTERNAL_MODE.lower() == "offline":
        dep_status["ncbi"] = "SKIPPED_OFFLINE"
    else:
        try:
            # Short timeout ping to NCBI einfo
            resp = httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
                params={"retmode": "json"},
                timeout=3.0,
            )
            if resp.status_code == 200:
                dep_status["ncbi"] = "UP"
            else:
                dep_status["ncbi"] = "DOWN"
                overall_ok = False
        except Exception as exc:
            logger.warning("Health check NCBI ping failure: %s", exc)
            dep_status["ncbi"] = "DOWN"
            overall_ok = False

    return {
        "status": "ok" if overall_ok else "degraded",
        "dependencies": dep_status,
    }
