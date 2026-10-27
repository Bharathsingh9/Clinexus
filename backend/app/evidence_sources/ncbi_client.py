"""NCBI E-utilities Client.

Specification Section 2, Section 4.3, Appendix A2, A9 (Req ID: RET-6, RET-7):
- HTTPS calls to ESearch and EFetch (db=pubmed).
- Rate limits: 3 requests/sec without API key, 10 requests/sec with API key.
- Bounded retries: up to 3 retries with exponential backoff (1s, 2s, 4s) + jitter.
- Timeout: configurable via settings.HTTP_TIMEOUT_S (15s).
- XML parsing into EvidenceDocument.
- abstract_missing flag when abstract is absent or empty.
- Publication types preserved as evidence_level.
- Offline mode: when EXTERNAL_MODE="offline", returns SKIPPED_OFFLINE without network calls.
- Per-query statuses: OK | EMPTY | TIMEOUT | RATE_LIMITED | MALFORMED | UNAVAILABLE | SKIPPED_OFFLINE.
"""

import logging
import os
import random
import threading
import time
from typing import Dict, List, Optional, Protocol, Tuple
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from backend.app.core.config import settings
from backend.app.schemas.evidence import EvidenceDocument, EvidenceSourceEnum, QueryStatusEnum

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token bucket rate limiter for NCBI API requests."""

    def __init__(self, max_per_second: float):
        self.interval = 1.0 / max(0.1, max_per_second)
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.monotonic()


class NCBIClientProtocol(Protocol):
    """Protocol for NCBI PubMed client."""

    def search_and_fetch(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> Tuple[List[EvidenceDocument], QueryStatusEnum]:
        """Execute ESearch then EFetch, returning parsed EvidenceDocument objects and query status."""
        ...


class NCBIClient:
    """Concrete HTTPS client for NCBI E-utilities."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        timeout_s: Optional[int] = None,
        max_retries: Optional[int] = None,
        offline: Optional[bool] = None,
    ):
        self.api_key = api_key or settings.NCBI_API_KEY
        self.email = email or settings.NCBI_EMAIL
        self.timeout_s = timeout_s or settings.HTTP_TIMEOUT_S
        self.max_retries = max_retries or settings.HTTP_MAX_RETRIES

        if offline is not None:
            self.offline = offline
        else:
            self.offline = (settings.EXTERNAL_MODE.lower() == "offline")

        # Rate limiting: 10 req/s with key, 3 req/s without key (SPEC 2, 4.3)
        rate_limit = 10.0 if self.api_key else 3.0
        self.rate_limiter = RateLimiter(max_per_second=rate_limit)

    def _execute_http_get(
        self,
        url: str,
        params: Dict[str, str],
    ) -> Tuple[Optional[httpx.Response], QueryStatusEnum]:
        """Execute rate-limited HTTP GET with exponential backoff and jitter."""
        if self.offline:
            return None, QueryStatusEnum.SKIPPED_OFFLINE

        # Rate limit enforcement
        self.rate_limiter.wait()

        # Add common params
        full_params = dict(params)
        if self.api_key:
            full_params["api_key"] = self.api_key
        if self.email:
            full_params["email"] = self.email

        backoff_delays = [1.0, 2.0, 4.0]

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=float(self.timeout_s)) as client:
                    resp = client.get(url, params=full_params)

                if resp.status_code == 200:
                    return resp, QueryStatusEnum.OK
                elif resp.status_code == 429:
                    logger.warning("NCBI rate limit hit (HTTP 429) on attempt %d", attempt + 1)
                    if attempt < self.max_retries:
                        delay = backoff_delays[min(attempt, len(backoff_delays) - 1)] + random.uniform(0.1, 0.5)
                        time.sleep(delay)
                        continue
                    return resp, QueryStatusEnum.RATE_LIMITED
                else:
                    logger.error("NCBI HTTP error status %d", resp.status_code)
                    if attempt < self.max_retries:
                        delay = backoff_delays[min(attempt, len(backoff_delays) - 1)] + random.uniform(0.1, 0.5)
                        time.sleep(delay)
                        continue
                    return resp, QueryStatusEnum.UNAVAILABLE

            except httpx.TimeoutException:
                logger.warning("NCBI request timed out on attempt %d", attempt + 1)
                if attempt < self.max_retries:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)] + random.uniform(0.1, 0.5)
                    time.sleep(delay)
                    continue
                return None, QueryStatusEnum.TIMEOUT

            except httpx.RequestError as exc:
                logger.warning("NCBI network error (%s) on attempt %d", exc, attempt + 1)
                if attempt < self.max_retries:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)] + random.uniform(0.1, 0.5)
                    time.sleep(delay)
                    continue
                return None, QueryStatusEnum.UNAVAILABLE

        return None, QueryStatusEnum.UNAVAILABLE

    def search_pubmed(
        self,
        query: str,
        max_results: int = 10,
    ) -> Tuple[List[str], QueryStatusEnum]:
        """Perform ESearch query against PubMed."""
        if self.offline:
            return [], QueryStatusEnum.SKIPPED_OFFLINE

        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results),
            "usehistory": "n",
        }

        resp, status = self._execute_http_get(self.ESEARCH_URL, params)
        if status != QueryStatusEnum.OK or resp is None:
            return [], status

        try:
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return [], QueryStatusEnum.EMPTY
            return id_list, QueryStatusEnum.OK
        except Exception as exc:
            logger.error("Failed to parse ESearch response: %s", exc)
            return [], QueryStatusEnum.MALFORMED

    def fetch_pubmed_xml(
        self,
        pmids: List[str],
    ) -> Tuple[Optional[str], QueryStatusEnum]:
        """Perform EFetch for list of PMIDs, returning raw XML string."""
        if self.offline:
            return None, QueryStatusEnum.SKIPPED_OFFLINE

        if not pmids:
            return None, QueryStatusEnum.EMPTY

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        resp, status = self._execute_http_get(self.EFETCH_URL, params)
        if status != QueryStatusEnum.OK or resp is None:
            return None, status

        return resp.text, QueryStatusEnum.OK

    def parse_pubmed_xml(
        self,
        xml_text: str,
    ) -> Tuple[List[EvidenceDocument], QueryStatusEnum]:
        """Parse PubmedArticleSet XML into EvidenceDocument models."""
        documents: List[EvidenceDocument] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as err:
            logger.error("NCBI XML parsing failed: %s", err)
            return [], QueryStatusEnum.MALFORMED

        for article_elem in root.findall(".//PubmedArticle"):
            medline = article_elem.find("MedlineCitation")
            if medline is None:
                continue

            # PMID
            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None
            if not pmid:
                continue

            article = medline.find("Article")

            # Title
            title = "UNKNOWN"
            if article is not None:
                t_elem = article.find("ArticleTitle")
                if t_elem is not None and t_elem.text:
                    title = "".join(t_elem.itertext()).strip()

            # Journal
            journal = "UNKNOWN"
            pub_date = "UNKNOWN"
            if article is not None:
                j_elem = article.find(".//Journal/Title")
                if j_elem is None or not (j_elem.text and j_elem.text.strip()):
                    j_elem = article.find(".//Journal/ISOAbbreviation")
                if j_elem is not None and j_elem.text:
                    journal = j_elem.text.strip()

                # Publication Date
                year_elem = article.find(".//JournalIssue/PubDate/Year")
                medline_date = article.find(".//JournalIssue/PubDate/MedlineDate")
                if year_elem is not None and year_elem.text:
                    pub_date = year_elem.text.strip()
                elif medline_date is not None and medline_date.text:
                    pub_date = medline_date.text.strip()[:4]

            # Abstract
            abstract_parts: List[str] = []
            if article is not None:
                for ab_elem in article.findall(".//Abstract/AbstractText"):
                    label = ab_elem.get("Label")
                    text = "".join(ab_elem.itertext()).strip()
                    if text:
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)

            abstract = " ".join(abstract_parts).strip() if abstract_parts else None
            abstract_missing = bool(not abstract)

            # DOI
            doi = None
            for id_elem in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if id_elem.get("IdType") == "doi" and id_elem.text:
                    doi = id_elem.text.strip()
                    break

            # Publication Types -> evidence_level (App A9)
            pub_types: List[str] = []
            if article is not None:
                for pt_elem in article.findall(".//PublicationTypeList/PublicationType"):
                    if pt_elem.text and pt_elem.text.strip():
                        pub_types.append(pt_elem.text.strip())

            evidence_level = pub_types[0] if pub_types else "UNKNOWN"

            doc = EvidenceDocument(
                source="PUBMED",
                source_id=pmid,
                title=title,
                authors=[],
                journal=journal,
                publication_date=pub_date,
                pmid=pmid,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                abstract=abstract,
                abstract_missing=abstract_missing,
                evidence_level=evidence_level,
            )
            documents.append(doc)

        status = QueryStatusEnum.OK if documents else QueryStatusEnum.EMPTY
        return documents, status

    def search_and_fetch(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> Tuple[List[EvidenceDocument], QueryStatusEnum]:
        """Execute full search then fetch pipeline."""
        if self.offline:
            return [], QueryStatusEnum.SKIPPED_OFFLINE

        limit = max_results or settings.EXTERNAL_MAX_RECORDS_PER_QUERY
        pmids, search_status = self.search_pubmed(query, max_results=limit)
        if search_status != QueryStatusEnum.OK or not pmids:
            return [], search_status

        xml_text, fetch_status = self.fetch_pubmed_xml(pmids)
        if fetch_status != QueryStatusEnum.OK or not xml_text:
            return [], fetch_status

        return self.parse_pubmed_xml(xml_text)
