"""API Router Aggregator.

Specification Section 7:
Mounts all 25 canonical endpoints defined in SPEC Section 7.
Prefix: /api/v1
"""

from fastapi import APIRouter

from backend.app.api.endpoints import (
    abstention,
    audit,
    auth,
    cases,
    evaluations,
    evidence,
    executions,
    health,
    reports,
    reviews,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(cases.router)
api_router.include_router(executions.router)
api_router.include_router(reports.router)
api_router.include_router(evidence.router)
api_router.include_router(abstention.router)
api_router.include_router(reviews.router)
api_router.include_router(audit.router)
api_router.include_router(evaluations.router)
