"""Fake NCBI Client for Unit and Integration Testing.

Satisfies NCBIClientProtocol for offline and synthetic testing.
"""

from typing import Dict, List, Optional, Tuple
from backend.app.schemas.evidence import EvidenceDocument, QueryStatusEnum


class FakeNCBIClient:
    """Deterministic fake NCBI E-utilities client for testing."""

    def __init__(
        self,
        canned_documents: Optional[Dict[str, List[EvidenceDocument]]] = None,
        default_status: QueryStatusEnum = QueryStatusEnum.OK,
        offline: bool = False,
    ):
        self.canned_documents: Dict[str, List[EvidenceDocument]] = canned_documents or {}
        self.default_status = default_status
        self.offline = offline
        self.calls: List[Dict[str, str]] = []

    def search_and_fetch(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> Tuple[List[EvidenceDocument], QueryStatusEnum]:
        """Return canned documents or simulated status for the query."""
        self.calls.append({"query": query, "max_results": str(max_results)})

        if self.offline:
            return [], QueryStatusEnum.SKIPPED_OFFLINE

        if query in self.canned_documents:
            docs = self.canned_documents[query]
            status = QueryStatusEnum.OK if docs else QueryStatusEnum.EMPTY
            return docs, status

        # Default fallback
        if self.default_status == QueryStatusEnum.OK:
            # Return a default synthetic document
            doc = EvidenceDocument(
                source="PUBMED",
                source_id="30001",
                title="[SYNTHETIC] Clinical presentation and diagnosis of respiratory conditions",
                authors=["Author A", "Author B"],
                journal="Journal of Synthetic Medicine",
                publication_date="2022",
                pmid="30001",
                doi="10.1000/synth.30001",
                url="https://pubmed.ncbi.nlm.nih.gov/30001/",
                abstract="[SYNTHETIC] Detailed clinical abstract describing acute presentation and diagnostic considerations.",
                abstract_missing=False,
                evidence_level="Review",
            )
            return [doc], QueryStatusEnum.OK

        return [], self.default_status
