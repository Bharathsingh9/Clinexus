"""Platform Dependency and Service Health Check Script.

Specification Section 7, Section 12, Rule R11.
Reports the real status of:
- Python runtime version
- Node runtime version
- PostgreSQL service status
- Qdrant vector database status
- Presence (SET/UNSET, never values) of LLM and NCBI credentials
"""

import os
import shutil
import subprocess
import sys

# Ensure backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.app.core.config import settings
except Exception as e:
    print(f"Error loading settings: {e}")
    sys.exit(1)


def check_python() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.executable})"


def check_node() -> str:
    node_bin = shutil.which("node")
    if not node_bin:
        return "NOT_FOUND"
    try:
        res = subprocess.run([node_bin, "--version"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.strip()
        return f"ERROR (code {res.returncode})"
    except Exception as e:
        return f"ERROR ({e})"


def check_postgres() -> str:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "HEALTHY (Connected successfully)"
    except Exception as e:
        return f"UNAVAILABLE ({type(e).__name__}: {e})"


def check_qdrant() -> str:
    try:
        import httpx
        url = settings.QDRANT_URL.rstrip("/") + "/readyz"
        resp = httpx.get(url, timeout=3.0)
        if resp.status_code == 200:
            return "HEALTHY (readyz 200 OK)"
        return f"UNHEALTHY (status {resp.status_code})"
    except Exception as e:
        return f"UNAVAILABLE ({type(e).__name__}: {e})"


def check_presence(val: str | None) -> str:
    if val and str(val).strip():
        return "PRESENT (configured)"
    return "NOT_CONFIGURED (unset)"


def main() -> None:
    print("==================================================")
    print("Clinexus Environment & Dependency Diagnostic Report")
    print("==================================================")
    print(f"Python Runtime:     {check_python()}")
    print(f"Node.js Runtime:    {check_node()}")
    print(f"PostgreSQL Service: {check_postgres()}")
    print(f"Qdrant Vector DB:   {check_qdrant()}")
    print("--------------------------------------------------")
    print("Configuration Presence (Values scrubbed per R11):")
    print(f"LLM_PROVIDER:       {settings.LLM_PROVIDER}")
    print(f"LLM_MODEL:          {settings.LLM_MODEL}")
    print(f"LLM_API_KEY:        {check_presence(settings.LLM_API_KEY)}")
    print(f"NCBI_EMAIL:         {check_presence(settings.NCBI_EMAIL)}")
    print(f"NCBI_API_KEY:       {check_presence(settings.NCBI_API_KEY)}")
    print(f"SECRET_KEY:         {check_presence(settings.SECRET_KEY)}")
    print(f"ADMIN_PASSWORD:     {check_presence(settings.ADMIN_PASSWORD)}")
    print("==================================================")


if __name__ == "__main__":
    main()
