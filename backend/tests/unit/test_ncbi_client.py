"""Unit tests for NCBI E-utilities Client (Req ID: RET-6, RET-7, App A9).

Specification Section 2, Section 4.3, Appendix A2, A9:
- Rate limiting parameters.
- Offline mode returns SKIPPED_OFFLINE and makes ZERO network calls.
- XML parsing into EvidenceDocument (PMID, DOI, abstract, title, date).
- abstract_missing flag when abstract element is missing or empty.
- Publication types preserved as evidence_level.
- Status values (OK, EMPTY, TIMEOUT, RATE_LIMITED, MALFORMED, UNAVAILABLE).
- Live network test marked "network" and excluded from default pytest run.
"""

from unittest.mock import MagicMock, patch
import pytest
from backend.app.evidence_sources.ncbi_client import NCBIClient, RateLimiter
from backend.app.schemas.evidence import QueryStatusEnum


def test_offline_mode_makes_zero_network_calls():
    """Verify EXTERNAL_MODE=offline makes zero network requests."""
    client = NCBIClient(offline=True)

    with patch("httpx.Client.get") as mock_get:
        docs, status = client.search_and_fetch("acute respiratory infection")
        assert status == QueryStatusEnum.SKIPPED_OFFLINE
        assert len(docs) == 0
        assert mock_get.call_count == 0


def test_ncbi_xml_parsing_complete_fields():
    """Verify XML parsing extracts title, PMID, DOI, abstract, date, and publication types as evidence_level."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation Status="MEDLINE" Owner="NLM">
          <PMID Version="1">31234567</PMID>
          <Article PubModel="Print">
            <Journal>
              <JournalIssue CitedMedium="Internet">
                <Volume>10</Volume>
                <Issue>2</Issue>
                <PubDate>
                  <Year>2023</Year>
                  <Month>May</Month>
                </PubDate>
              </JournalIssue>
              <Title>New England Journal of Clinical Science</Title>
              <ISOAbbreviation>N Engl J Clin Sci</ISOAbbreviation>
            </Journal>
            <ArticleTitle>Clinical assessment of acute chest pain syndromes.</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">Chest pain represents a frequent chief complaint in acute care.</AbstractText>
              <AbstractText Label="METHODS">Diagnostic evaluation was conducted across cohorts.</AbstractText>
              <AbstractText Label="CONCLUSIONS">Systematic workup identifies critical presentations.</AbstractText>
            </Abstract>
            <PublicationTypeList>
              <PublicationType UI="D016428">Journal Article</PublicationType>
              <PublicationType UI="D016454">Review</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">31234567</ArticleId>
            <ArticleId IdType="doi">10.1056/NEJMoa31234567</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NCBIClient(offline=True)
    docs, status = client.parse_pubmed_xml(sample_xml)

    assert status == QueryStatusEnum.OK
    assert len(docs) == 1
    doc = docs[0]

    assert doc.pmid == "31234567"
    assert doc.doi == "10.1056/NEJMoa31234567"
    assert doc.title == "Clinical assessment of acute chest pain syndromes."
    assert "Chest pain represents a frequent chief complaint" in (doc.abstract or "")
    assert "BACKGROUND:" in (doc.abstract or "")
    assert doc.abstract_missing is False
    assert doc.publication_date == "2023"
    assert doc.journal == "New England Journal of Clinical Science"
    assert doc.evidence_level == "Journal Article"  # First publication type stored (App A9)


def test_ncbi_xml_parsing_missing_abstract_flag():
    """Verify abstract_missing flag is True when <Abstract> is omitted."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>99887766</PMID>
          <Article>
            <ArticleTitle>Editorial without abstract</ArticleTitle>
            <Journal>
              <Title>BMJ</Title>
            </Journal>
            <PublicationTypeList>
              <PublicationType>Editorial</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    client = NCBIClient(offline=True)
    docs, status = client.parse_pubmed_xml(sample_xml)

    assert status == QueryStatusEnum.OK
    assert len(docs) == 1
    doc = docs[0]
    assert doc.pmid == "99887766"
    assert doc.abstract is None
    assert doc.abstract_missing is True
    assert doc.evidence_level == "Editorial"


def test_ncbi_rate_limiter_interval():
    """Verify RateLimiter enforces minimum intervals."""
    limiter = RateLimiter(max_per_second=10.0)
    assert limiter.interval == 0.1
    limiter_slow = RateLimiter(max_per_second=3.0)
    assert round(limiter_slow.interval, 3) == 0.333


@pytest.mark.network
def test_live_ncbi_smoke():
    """Live smoke test against official NCBI E-utilities API.
    Excluded from default test run; carries 'network' marker.
    """
    client = NCBIClient(offline=False)
    docs, status = client.search_and_fetch("asthma diagnosis", max_results=2)
    assert status in (QueryStatusEnum.OK, QueryStatusEnum.EMPTY, QueryStatusEnum.RATE_LIMITED)
