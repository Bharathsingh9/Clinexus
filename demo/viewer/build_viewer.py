"""Build a self-contained demo/viewer/index.html from demo/viewer/data/results.json.

Reads results.json produced by export_demo_results.py and writes ONE self-contained
HTML file: inline CSS, inline JavaScript, JSON data embedded in a <script> tag.
No network requests, no external fonts or CDNs. Works by double-clicking.

Authorization: One-off demo viewer exception DEC-041 (docs/decision_log.md).
Scope: demo/viewer/ only. No application code is modified.
"""

import json
import sys
from pathlib import Path

VIEWER_DIR = Path(__file__).parent
DATA_PATH = VIEWER_DIR / "data" / "results.json"
OUTPUT_PATH = VIEWER_DIR / "index.html"


def load_results() -> dict:
    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} not found.")
        print("  Run: python demo/viewer/export_demo_results.py")
        sys.exit(1)
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def status_color_class(status: str) -> str:
    """Map workflow/node status to CSS class name."""
    s = (status or "").upper()
    if s in ("COMPLETED", "READY_FOR_REVIEW", "REVIEWED"):
        return "status-blue"
    if s in ("ABSTAINED", "FAILED", "CONTRADICTED", "PARTIALLY_SUPPORTED"):
        return "status-red"
    if s in ("RUNNING", "PARTIALLY_SUPPORTED", "DEGRADED", "INSUFFICIENT_EVIDENCE"):
        return "status-amber"
    if s in ("ABSTAINED",):
        return "status-purple"
    return "status-amber"


def gate_outcome_class(outcome: str) -> str:
    o = (outcome or "").lower()
    if "ready" in o:
        return "status-blue"
    if "abstain" in o:
        return "status-purple"
    return "status-amber"


def build_html(data: dict) -> str:
    generated_at = data.get("generated_at", "unknown")
    honesty_banner = data.get("honesty_banner", "")
    scenarios = data.get("scenarios", [])
    db = data.get("database", "clinexus_demo")
    coll_ver = data.get("qdrant_collection_version", "777")

    # Embed JSON for JavaScript access
    embedded_json = json.dumps(data, indent=None, separators=(",", ":"), default=str)
    # Escape </script> inside JSON string to prevent early termination
    embedded_json = embedded_json.replace("</script>", "<\\/script>")

    # Build tab headers
    tab_headers = ""
    for s in scenarios:
        sid = s["scenario"]
        label = f"Scenario {sid}"
        tab_headers += (
            f'<button class="tab-btn" id="tab-btn-{sid}" '
            f'onclick="showScenario(\'{sid}\')" '
            f'aria-label="{label}">{label}</button>\n'
        )

    # Build scenario panels
    scenario_panels = ""
    for s in scenarios:
        sid = s["scenario"]
        title = s.get("title", "")
        elapsed = s.get("elapsed_s", 0)
        final_state = s.get("final_state", s.get("workflow_status", ""))
        urgent_banner = s.get("urgent_banner") or ""
        urgent_pre = s.get("urgent_pre_analysis", {})

        # Status class for final state
        fsc = status_color_class(final_state)

        # Urgent alert
        urgent_html = ""
        if urgent_banner:
            urgent_html = f"""
            <div class="urgent-alert" role="alert">
              <strong>⚠ URGENT SYMPTOM ALERT</strong><br>
              {_esc(urgent_banner)}<br>
              <span class="tag-real">REAL — pre-analysis pattern match fires before workflow</span>
            </div>"""
        if urgent_pre.get("is_urgent"):
            urgent_html += f"""
            <div class="urgent-alert" role="alert">
              <strong>Pre-Analysis Safety Check (REAL):</strong>
              Urgent pattern matched: {_esc(", ".join(urgent_pre.get("matched_rule_names", [])))}
            </div>"""

        # Clinical note for scenario C
        clinical_note_html = ""
        if s.get("clinical_note"):
            clinical_note_html = (
                f'<div class="clinical-note">{_esc(s["clinical_note"])}</div>'
            )

        # Pipeline timeline
        timeline_html = _build_timeline(s.get("timeline", []))

        # Gate panel
        gate_html = _build_gate_panel(s.get("gate"), s.get("abstention"))

        # Abstention panel
        abstention_html = _build_abstention_panel(s.get("abstention"))

        # Claims table
        claims_html = _build_claims_table(
            s.get("claims", []), s.get("retrieved_evidence", [])
        )

        # Provenance panel (scenario A only)
        provenance_html = ""
        if s.get("provenance_edges"):
            provenance_html = _build_provenance_panel(s["provenance_edges"])

        # Optimizer iterations panel (scenario B only)
        iterations_html = ""
        if s.get("iterations"):
            iterations_html = _build_iterations_panel(s["iterations"])

        # Numeric veto panel (scenario B only)
        veto_html = ""
        if s.get("numeric_veto_demonstration"):
            veto_html = _build_veto_panel(s["numeric_veto_demonstration"])

        # Review / Audit panel (scenario D only)
        review_html = ""
        if sid == "D":
            review_html = _build_review_panel(s)

        # Origin summary
        origin_html = _build_origin_summary(s.get("_origin_summary", {}))

        scenario_panels += f"""
        <div class="scenario-panel" id="panel-{sid}" role="tabpanel" aria-labelledby="tab-btn-{sid}" hidden>
          <h2 class="scenario-title">{_esc(title)}</h2>
          <div class="meta-row">
            <span>Database: <code>{_esc(db)}</code> &nbsp;|&nbsp; Qdrant v{_esc(str(coll_ver))}</span>
            <span>Elapsed: <strong>{elapsed:.3f}s</strong></span>
            <span>Final State: <span class="status-badge {fsc}">{_esc(final_state)}</span></span>
          </div>
          {urgent_html}
          {clinical_note_html}
          <section aria-label="Pipeline Timeline">
            <h3>Pipeline Timeline <span class="tag-real">REAL</span></h3>
            {timeline_html}
          </section>
          {gate_html}
          {abstention_html}
          {claims_html}
          {provenance_html}
          {iterations_html}
          {veto_html}
          {review_html}
          {origin_html}
        </div>
        """

    css = _build_css()
    js = _build_js()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clinexus Demo Viewer — Research Prototype</title>
  <style>
{css}
  </style>
</head>
<body>
  <!-- Permanent honesty banner -->
  <div class="honesty-banner" role="banner" aria-live="polite">
    <strong>DEMO VIEWER: not the Clinexus application interface.</strong>
    Language-model outputs are <strong>SCRIPTED</strong>;
    embeddings, retrieval, workflow, verifier checks, gate, provenance, and audit are <strong>REAL</strong>.
    Research prototype, not clinically validated. No accuracy results are shown.
  </div>

  <header class="page-header">
    <h1>Clinexus Clinical Decision Support<br><span class="subtitle">Live Workflow Demonstration Viewer</span></h1>
    <p class="generated-at">Generated: {_esc(generated_at)} &nbsp;|&nbsp; Authorization: DEC-041</p>
  </header>

  <!-- Legend -->
  <div class="legend" role="note" aria-label="Color legend">
    <span class="legend-item"><span class="dot status-blue"></span> Completed / Ready</span>
    <span class="legend-item"><span class="dot status-amber"></span> Partial / Running / Unverified</span>
    <span class="legend-item"><span class="dot status-red"></span> Contradicted / Failed / Vetoed</span>
    <span class="legend-item"><span class="dot status-purple"></span> Abstained</span>
    <span class="legend-item"><span class="tag-real">REAL</span> Implemented system component</span>
    <span class="legend-item"><span class="tag-scripted">SCRIPTED</span> Demo stand-in</span>
    <span class="legend-item"><span class="tag-human">HUMAN</span> Clinician-authored content</span>
  </div>

  <!-- Scenario tabs -->
  <nav class="tab-bar" role="tablist" aria-label="Demo Scenarios">
    {tab_headers}
  </nav>

  <main class="main-content">
    {scenario_panels}
  </main>

  <footer class="page-footer">
    <p>Clinexus Demo Viewer &mdash; Research Prototype &mdash; Not for clinical use &mdash; DEC-041</p>
    <p>No accuracy, recall, or benchmark numbers are displayed. Retrieval output is illustrative, subset corpus, unconfirmed labels, not an evaluation result.</p>
  </footer>

  <script>
  // Embedded scenario data (origin: export_demo_results.py run against clinexus_demo)
  const DEMO_DATA = {embedded_json};

  {js}

  // Initialise on load
  document.addEventListener("DOMContentLoaded", function() {{
    const firstId = DEMO_DATA.scenarios.length > 0 ? DEMO_DATA.scenarios[0].scenario : "A";
    showScenario(firstId);
  }});
  </script>
</body>
</html>"""
    return html


def _esc(s: str) -> str:
    """HTML-escape a string."""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return (s
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def _build_timeline(timeline: list) -> str:
    if not timeline:
        return "<p class='empty-note'>No timeline data.</p>"

    # Horizontal pipeline
    pipes = ""
    for i, node in enumerate(timeline):
        name = node.get("node", "?")
        status = node.get("status", "UNKNOWN")
        dur = node.get("duration_ms")
        dur_str = f"{dur} ms" if dur is not None else "< 10 ms"
        note = node.get("note", "")
        sc = status_color_class(status)
        arrow = " &rarr; " if i < len(timeline) - 1 else ""
        pipes += f"""<span class="pipe-node {sc}" tabindex="0"
          aria-label="{_esc(name)}: {_esc(status)}, {_esc(dur_str)}"
          title="{_esc(note)}">{_esc(name)}<br><small>{_esc(status)}</small><br><small>{_esc(dur_str)}</small></span>{arrow}"""

    # Table
    rows = ""
    for node in timeline:
        name = node.get("node", "?")
        status = node.get("status", "UNKNOWN")
        dur = node.get("duration_ms")
        dur_str = f"{dur} ms" if dur is not None else "< 10 ms"
        note = node.get("note", "")
        sc = status_color_class(status)
        rows += f"""<tr>
          <td>{_esc(name)}</td>
          <td><span class="status-badge {sc}">{_esc(status)}</span></td>
          <td>{_esc(dur_str)}</td>
          <td class="note-cell">{_esc(note[:100])}</td>
        </tr>"""

    return f"""
    <div class="pipeline" role="list" aria-label="Pipeline steps">
      {pipes}
    </div>
    <details class="timeline-table-details">
      <summary>Full timeline table</summary>
      <table class="data-table" aria-label="Timeline details">
        <thead><tr><th>Node</th><th>Status</th><th>Duration</th><th>Note</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </details>"""


def _build_gate_panel(gate: dict, abstention: dict) -> str:
    if not gate:
        return ""
    outcome = gate.get("outcome", "")
    reason_codes = gate.get("reason_codes", [])
    explanation = gate.get("explanation", "")
    rule_triggered = gate.get("rule_triggered")
    oc = gate_outcome_class(outcome)

    rule_str = f"Rule {rule_triggered}" if rule_triggered else "All rules evaluated"
    codes_html = ", ".join(f"<code>{_esc(c)}</code>" for c in reason_codes) if reason_codes else "<em>None</em>"

    return f"""
    <section aria-label="Evidence Safety Gate">
      <h3>Evidence Safety Gate <span class="tag-real">REAL</span></h3>
      <div class="panel gate-panel">
        <div class="gate-outcome">
          <span class="status-badge {oc}">{_esc(outcome)}</span>
          <span class="gate-rule">{_esc(rule_str)}</span>
        </div>
        <p><strong>Reason codes:</strong> {codes_html}</p>
        <p><strong>Explanation:</strong> {_esc(explanation)}</p>
      </div>
    </section>"""


def _build_abstention_panel(abstention: dict) -> str:
    if not abstention:
        return ""
    reason_codes = abstention.get("reason_codes", [])
    explanation = abstention.get("explanation", "")
    affected = abstention.get("affected_sections", [])
    steps = abstention.get("suggested_next_steps", [])

    codes_html = "".join(f"<li><code>{_esc(c)}</code></li>" for c in reason_codes)
    steps_html = "".join(f"<li>{_esc(s)}</li>" for s in steps) if steps else ""
    affected_html = ", ".join(f"<code>{_esc(a)}</code>" for a in affected) if affected else "<em>None</em>"

    return f"""
    <section aria-label="Abstention Decision">
      <h3>Abstention Decision <span class="tag-real">REAL</span> <span class="status-badge status-purple">ABSTAINED</span></h3>
      <div class="panel abstention-panel">
        <p><strong>Explanation:</strong> {_esc(explanation)}</p>
        <p><strong>Reason Codes:</strong></p>
        <ul class="reason-codes">{codes_html}</ul>
        <p><strong>Affected Sections:</strong> {affected_html}</p>
        {"<p><strong>Suggested Next Steps:</strong></p><ul>" + steps_html + "</ul>" if steps_html else ""}
      </div>
    </section>"""


def _build_claims_table(claims: list, retrieved_evidence: list) -> str:
    if not claims:
        return """
    <section aria-label="Claims">
      <h3>Claims <span class="tag-scripted">SCRIPTED text</span> / <span class="tag-real">REAL verification</span></h3>
      <p class="empty-note">No claims generated (abstained before claim synthesis or no relevant evidence).</p>
    </section>"""

    # Build evidence lookup for previews
    ev_map = {e["evidence_id"]: e for e in retrieved_evidence}

    rows = ""
    for claim in claims:
        cid = claim.get("claim_id", "")
        text = claim.get("claim_text", "")
        section = claim.get("section", "")
        vstat = claim.get("verification_status") or "UNKNOWN"
        vjust = claim.get("verification_justification") or ""
        vmethod = claim.get("verification_method") or ""
        vlim = claim.get("verification_limitations") or ""
        cited = claim.get("cited_evidence_ids", [])
        vc = status_color_class(vstat)

        # Evidence previews
        ev_rows = ""
        for eid in cited:
            ev = ev_map.get(eid, {})
            _score = ev.get("retrieval_score", "")
            _score_str = f"{_score:.3f}" if isinstance(_score, float) else _esc(str(_score))
            ev_rows += f"""
              <tr>
                <td><code>{_esc(eid)}</code></td>
                <td>{_esc(ev.get("source", ""))}</td>
                <td class="text-preview">{_esc(ev.get("text_preview", "")[:180])}</td>
                <td>{_score_str}</td>
              </tr>"""

        # If PARTIALLY_SUPPORTED or not SUPPORTED → show no-evidence note
        no_evidence_note = ""
        if vstat in ("PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE", "CONTRADICTED", "UNVERIFIABLE"):
            no_evidence_note = '<p class="no-evidence-note">⚠ No fully supporting evidence found. Claim downgraded by real deterministic verification.</p>'
        if vstat == "UNKNOWN" or vstat not in ("SUPPORTED",):
            no_evidence_note = no_evidence_note or '<p class="no-evidence-note">Verification status unknown or not supported.</p>'

        lim_html = f'<p class="limitation"><em>Limitation: {_esc(vlim)}</em></p>' if vlim else ""

        rows += f"""<tr class="claim-row" tabindex="0" onclick="toggleClaimDetail('{_esc(cid)}')"
            onkeydown="if(event.key==='Enter'||event.key===' ')toggleClaimDetail('{_esc(cid)}')"
            aria-expanded="false" aria-controls="claim-detail-{_esc(cid)}">
          <td class="claim-text-cell">
            <span class="claim-text">{_esc(text[:120])}{"…" if len(text) > 120 else ""}</span>
            <span class="tag-scripted">SCRIPTED text</span>
          </td>
          <td>{_esc(section)}</td>
          <td><span class="status-badge {vc}">{_esc(vstat)}</span> <span class="tag-real">REAL</span></td>
        </tr>
        <tr class="claim-detail" id="claim-detail-{_esc(cid)}" hidden>
          <td colspan="3">
            <div class="claim-detail-inner">
              <p><strong>Full claim text <span class="tag-scripted">SCRIPTED</span>:</strong> {_esc(text)}</p>
              <p><strong>Verification method <span class="tag-real">REAL</span>:</strong> {_esc(vmethod)}</p>
              <p><strong>Justification <span class="tag-real">REAL</span>:</strong> {_esc(vjust)}</p>
              {lim_html}
              {no_evidence_note}
              {"<h4>Cited Evidence Passages <span class='tag-real'>REAL retrieval</span></h4><table class='evidence-table'><thead><tr><th>Evidence ID</th><th>Source</th><th>Text Preview</th><th>Score</th></tr></thead><tbody>" + ev_rows + "</tbody></table>" if ev_rows else "<p><em>No cited passages available in retrieved set.</em></p>"}
            </div>
          </td>
        </tr>"""

    return f"""
    <section aria-label="Claims Table">
      <h3>Claims <span class="tag-scripted">SCRIPTED text</span> / <span class="tag-real">REAL verification</span></h3>
      <p class="instruction">Click a row to expand evidence and verification details.</p>
      <table class="data-table claims-table" aria-label="Claims">
        <thead><tr><th>Claim Text</th><th>Section</th><th>Verification Status</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def _build_provenance_panel(edges: list) -> str:
    if not edges:
        return ""
    rows = ""
    for e in edges[:12]:
        rows += f"""<tr>
          <td><code>{_esc(e.get("relation", ""))}</code></td>
          <td><small><code>{_esc((e.get("claim_id") or "")[:16])}…</code></small></td>
          <td><small><code>{_esc((e.get("evidence_chunk_id") or e.get("document_id") or "")[:16])}…</code></small></td>
        </tr>"""
    total = len(edges)
    return f"""
    <section aria-label="Provenance Graph">
      <h3>Claim-to-Evidence Provenance (SPEC 5.3) <span class="tag-real">REAL</span></h3>
      <p>Total relational provenance edges recorded: <strong>{total}</strong></p>
      <table class="data-table" aria-label="Provenance edges">
        <thead><tr><th>Relation</th><th>Claim ID (prefix)</th><th>Target ID (prefix)</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      {"<p><em>Showing first 12 of " + str(total) + " edges.</em></p>" if total > 12 else ""}
    </section>"""


def _build_iterations_panel(iterations: list) -> str:
    if not iterations:
        return ""
    rows = ""
    for it in iterations:
        rows += f"""<tr>
          <td>{it.get("iteration_number", "")}</td>
          <td><code>{_esc(it.get("node_selected", ""))}</code></td>
          <td>{it.get("confidence_before", 0):.3f}</td>
          <td>{it.get("confidence_after", "—") if it.get("confidence_after") is None else f"{it['confidence_after']:.3f}"}</td>
          <td>{_esc(it.get("action_taken", ""))}</td>
        </tr>"""
    return f"""
    <section aria-label="Optimizer Iterations">
      <h3>Adaptive Optimizer Iterations <span class="tag-real">REAL</span></h3>
      <table class="data-table" aria-label="Optimizer iterations">
        <thead><tr><th>#</th><th>Node Selected</th><th>Conf Before</th><th>Conf After</th><th>Action</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def _build_veto_panel(veto: dict) -> str:
    scripted_out = veto.get("scripted_judge_output", "")
    real_result = veto.get("real_code_veto_result", "")
    reason = veto.get("veto_reason", "")
    claim = veto.get("veto_claim") or {}
    claim_text = claim.get("claim_text", "")
    veto_status = claim.get("status", "")
    vc = status_color_class(veto_status)

    return f"""
    <section aria-label="Numeric Veto Demonstration">
      <h3>Deterministic Numeric Veto Demonstration</h3>
      <div class="panel veto-panel">
        <div class="veto-row">
          <div class="veto-col">
            <span class="tag-scripted">SCRIPTED judge output</span>
            <span class="status-badge status-blue">{_esc(scripted_out)}</span>
          </div>
          <div class="veto-arrow">&rarr; real code veto &rarr;</div>
          <div class="veto-col">
            <span class="tag-real">REAL code result</span>
            <span class="status-badge {vc}">{_esc(real_result)}</span>
          </div>
        </div>
        <p><strong>Veto reason:</strong> {_esc(reason)}</p>
        <p><strong>Claim text (SCRIPTED):</strong> <em>{_esc(claim_text[:180])}</em></p>
        <p class="no-evidence-note">⚠ The number "99" injected by the scripted model does not appear in any cited passage.
        Real deterministic Python regex veto catches this and downgrades to {_esc(real_result)}, overriding the scripted judge.</p>
      </div>
    </section>"""


def _build_review_panel(s: dict) -> str:
    rbac = s.get("rbac_rejection", {})
    clinician_review = s.get("clinician_review", {})
    append_only = s.get("append_only_verification", {})
    invalid_trans = s.get("invalid_transition", {})
    audit_trail = s.get("audit_trail", [])

    # RBAC section
    rbac_html = f"""
    <div class="panel rbac-panel">
      <h4>Step D.1 — RBAC Rejection <span class="tag-real">REAL</span></h4>
      <p>Role <code>viewer</code> attempted <code>{_esc(rbac.get("attempted_action",""))}</code>:</p>
      <p class="no-evidence-note">⛔ {_esc(rbac.get("rejection_message",""))}</p>
    </div>"""

    # Clinician review section
    cr = clinician_review
    cr_html = f"""
    <div class="panel review-panel">
      <h4>Step D.2 — Clinician Review <span class="tag-real">REAL</span></h4>
      <p>Action: <code>{_esc(cr.get("action",""))}</code> &nbsp;|&nbsp;
         Outcome: <strong>{_esc(cr.get("outcome",""))}</strong></p>
      <div class="human-note">
        <span class="tag-human">HUMAN</span>
        <span class="human-note-text">{_esc(cr.get("note",""))}</span>
      </div>
      <p>Case state after review: <span class="status-badge status-blue">{_esc(cr.get("case_status_after",""))}</span></p>
    </div>"""

    # Append-only verification
    reviews = append_only.get("reviews", [])
    rev_rows = ""
    for idx, r in enumerate(reviews, 1):
        is_human = r.get("is_human", False)
        human_tag = '<span class="tag-human">HUMAN</span>' if is_human else ""
        note_html = f'<div class="human-note">{human_tag} {_esc(r.get("note",""))}</div>' if r.get("note") else ""
        rev_rows += f"""<tr>
          <td>#{idx}</td>
          <td><code>{_esc(r.get("action",""))}</code></td>
          <td>{_esc(r.get("outcome",""))}</td>
          <td>{note_html}</td>
        </tr>"""

    ao_html = f"""
    <div class="panel">
      <h4>Step D.3 — Append-Only Persistence <span class="tag-real">REAL</span></h4>
      <p>Total reviews recorded: <strong>{append_only.get("total_reviews","")}</strong> — earlier review preserved, not overwritten.</p>
      <table class="data-table" aria-label="Review records">
        <thead><tr><th>#</th><th>Action</th><th>Outcome</th><th>Note</th></tr></thead>
        <tbody>{rev_rows}</tbody>
      </table>
    </div>"""

    # Invalid transition
    it_html = f"""
    <div class="panel">
      <h4>Step D.4 — Invalid State Transition Rejection <span class="tag-real">REAL</span></h4>
      <p>Attempted: <code>{_esc(invalid_trans.get("attempted",""))}</code></p>
      <p class="no-evidence-note">⛔ {_esc(invalid_trans.get("rejection_message",""))}</p>
      <p><em>{_esc(invalid_trans.get("note",""))}</em></p>
    </div>"""

    # Audit trail
    audit_rows = ""
    for evt in audit_trail:
        is_human = evt.get("is_human", False)
        human_tag = '<span class="tag-human">HUMAN</span>' if is_human else ""
        payload = evt.get("payload", {})
        payload_str = json.dumps(payload, default=str)[:80] + ("…" if len(json.dumps(payload, default=str)) > 80 else "")
        ts = evt.get("timestamp", "")[:19].replace("T", " ")
        audit_rows += f"""<tr {"class='human-row'" if is_human else ""}>
          <td><small>{_esc(ts)}</small></td>
          <td><code>{_esc(evt.get("event_type",""))}</code></td>
          <td>{_esc(evt.get("actor_role",""))}</td>
          <td>{_esc(evt.get("resource_type",""))}</td>
          <td><small>{_esc(payload_str)}</small> {human_tag}</td>
        </tr>"""

    audit_html = f"""
    <div class="panel">
      <h4>Immutable Audit Trail <span class="tag-real">REAL</span></h4>
      <p>Total audit events: <strong>{len(audit_trail)}</strong>.
         <span class="tag-human">HUMAN</span> rows are clinician-authored events.</p>
      <table class="data-table audit-table" aria-label="Audit trail">
        <thead><tr><th>Timestamp</th><th>Event Type</th><th>Actor Role</th><th>Resource</th><th>Payload</th></tr></thead>
        <tbody>{audit_rows}</tbody>
      </table>
    </div>"""

    return f"""
    <section aria-label="HITL Review and Audit">
      <h3>HITL Review &amp; Audit <span class="tag-real">REAL</span></h3>
      {rbac_html}
      {cr_html}
      {ao_html}
      {it_html}
      {audit_html}
    </section>"""


def _build_origin_summary(origin_summary: dict) -> str:
    if not origin_summary:
        return ""
    rows = ""
    for component, origin in origin_summary.items():
        if origin == "REAL":
            tag = '<span class="tag-real">REAL</span>'
        elif origin == "SCRIPTED":
            tag = '<span class="tag-scripted">SCRIPTED</span>'
        elif "SYNTHETIC" in origin:
            tag = f'<span class="tag-scripted">{_esc(origin)}</span>'
        else:
            tag = f'<code>{_esc(origin)}</code>'
        rows += f"<tr><td>{_esc(component)}</td><td>{tag}</td></tr>"
    return f"""
    <section aria-label="Origin Summary">
      <h3>What Was Real vs Scripted (This Scenario)</h3>
      <table class="data-table origin-table" aria-label="Origin summary">
        <thead><tr><th>Component</th><th>Origin</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def _build_css() -> str:
    return """
    :root {
      --bg: #ffffff;
      --bg-alt: #f3f4f6;
      --bg-panel: #f9fafb;
      --border: #d1d5db;
      --text: #111827;
      --text-muted: #6b7280;
      --blue: #1d4ed8;
      --blue-bg: #eff6ff;
      --blue-border: #bfdbfe;
      --amber: #b45309;
      --amber-bg: #fffbeb;
      --amber-border: #fde68a;
      --red: #b91c1c;
      --red-bg: #fef2f2;
      --red-border: #fecaca;
      --purple: #7c3aed;
      --purple-bg: #f5f3ff;
      --purple-border: #ddd6fe;
      --urgent: #7f1d1d;
      --urgent-bg: #fee2e2;
      --human-bg: #f0fdf4;
      --human-border: #bbf7d0;
      --human-text: #14532d;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111827;
        --bg-alt: #1f2937;
        --bg-panel: #1f2937;
        --border: #374151;
        --text: #f9fafb;
        --text-muted: #9ca3af;
        --blue: #60a5fa;
        --blue-bg: #1e3a5f;
        --blue-border: #1e40af;
        --amber: #fbbf24;
        --amber-bg: #451a03;
        --amber-border: #92400e;
        --red: #f87171;
        --red-bg: #450a0a;
        --red-border: #7f1d1d;
        --purple: #c084fc;
        --purple-bg: #2e1065;
        --purple-border: #6b21a8;
        --urgent: #fca5a5;
        --urgent-bg: #450a0a;
        --human-bg: #052e16;
        --human-border: #166534;
        --human-text: #86efac;
      }
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font);
      font-size: 17px;
      line-height: 1.6;
      color: var(--text);
      background: var(--bg);
      max-width: 1200px;
      margin: 0 auto;
      padding: 0 1.5rem 3rem;
    }

    .honesty-banner {
      background: var(--amber-bg);
      border: 2px solid var(--amber-border);
      color: var(--amber);
      padding: 0.9rem 1.4rem;
      margin: 1rem 0;
      border-radius: 6px;
      font-size: 0.97rem;
    }

    .page-header { margin: 1.5rem 0 1rem; }
    .page-header h1 { font-size: 1.7rem; font-weight: 700; }
    .page-header .subtitle { font-size: 1.1rem; font-weight: 400; color: var(--text-muted); }
    .generated-at { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.3rem; }

    .legend {
      display: flex; flex-wrap: wrap; gap: 0.8rem 1.4rem;
      padding: 0.7rem 1rem; background: var(--bg-alt);
      border: 1px solid var(--border); border-radius: 6px;
      margin-bottom: 1.2rem; font-size: 0.88rem;
      align-items: center;
    }
    .legend-item { display: flex; align-items: center; gap: 0.35rem; }
    .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; }

    .tab-bar {
      display: flex; flex-wrap: wrap; gap: 0.4rem;
      border-bottom: 2px solid var(--border); margin-bottom: 1.5rem;
      padding-bottom: 0;
    }
    .tab-btn {
      padding: 0.55rem 1.3rem; cursor: pointer;
      border: 1px solid var(--border); border-bottom: none;
      border-radius: 6px 6px 0 0;
      background: var(--bg-alt); color: var(--text);
      font-size: 1rem; font-family: var(--font);
      transition: background 0.15s;
    }
    .tab-btn:hover { background: var(--blue-bg); color: var(--blue); }
    .tab-btn.active { background: var(--bg); border-bottom: 2px solid var(--bg); font-weight: 600; color: var(--blue); }
    .tab-btn:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }

    .scenario-panel { animation: fadein 0.2s ease; }
    @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }

    .scenario-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.7rem; }
    .meta-row { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; font-size: 0.92rem; color: var(--text-muted); }

    section { margin: 1.8rem 0; }
    section h3 { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }

    .panel {
      background: var(--bg-panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .gate-panel { border-left: 4px solid var(--blue); }
    .abstention-panel { border-left: 4px solid var(--purple); }
    .veto-panel { border-left: 4px solid var(--red); }
    .rbac-panel { border-left: 4px solid var(--red); }
    .review-panel { border-left: 4px solid var(--blue); }

    .gate-outcome { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.7rem; }
    .gate-rule { color: var(--text-muted); font-size: 0.9rem; }

    .veto-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
    .veto-col { display: flex; flex-direction: column; gap: 0.3rem; align-items: flex-start; }
    .veto-arrow { font-weight: 600; color: var(--text-muted); }

    .status-badge {
      display: inline-block; padding: 0.2rem 0.6rem;
      border-radius: 4px; font-size: 0.82rem; font-weight: 600;
    }
    .status-blue { background: var(--blue-bg); color: var(--blue); border: 1px solid var(--blue-border); }
    .status-amber { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-border); }
    .status-red { background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }
    .status-purple { background: var(--purple-bg); color: var(--purple); border: 1px solid var(--purple-border); }

    .dot.status-blue { background: var(--blue); }
    .dot.status-amber { background: var(--amber); }
    .dot.status-red { background: var(--red); }
    .dot.status-purple { background: var(--purple); }

    .tag-real, .tag-scripted, .tag-human {
      display: inline-block; padding: 0.1rem 0.45rem;
      border-radius: 3px; font-size: 0.75rem; font-weight: 700;
      letter-spacing: 0.02em; text-transform: uppercase; vertical-align: middle;
    }
    .tag-real { background: var(--blue-bg); color: var(--blue); border: 1px solid var(--blue-border); }
    .tag-scripted { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber-border); }
    .tag-human { background: var(--human-bg); color: var(--human-text); border: 1px solid var(--human-border); }

    .urgent-alert {
      background: var(--urgent-bg); color: var(--urgent);
      border: 2px solid #dc2626; border-radius: 6px;
      padding: 0.8rem 1.1rem; margin-bottom: 1rem;
      font-size: 0.95rem;
    }

    .clinical-note {
      background: var(--amber-bg); border: 1px solid var(--amber-border);
      border-radius: 6px; padding: 0.7rem 1rem; margin-bottom: 1rem;
      font-size: 0.9rem; color: var(--amber);
    }

    .pipeline {
      display: flex; flex-wrap: wrap; align-items: center;
      gap: 0.3rem; margin-bottom: 1rem; overflow-x: auto;
    }
    .pipe-node {
      background: var(--bg-alt); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.4rem 0.7rem;
      font-size: 0.8rem; text-align: center;
      cursor: default; transition: box-shadow 0.15s;
    }
    .pipe-node:focus-visible { outline: 3px solid var(--blue); outline-offset: 2px; }
    .pipe-node.status-blue { background: var(--blue-bg); border-color: var(--blue-border); color: var(--blue); }
    .pipe-node.status-amber { background: var(--amber-bg); border-color: var(--amber-border); color: var(--amber); }
    .pipe-node.status-red { background: var(--red-bg); border-color: var(--red-border); color: var(--red); }
    .pipe-node.status-purple { background: var(--purple-bg); border-color: var(--purple-border); color: var(--purple); }

    .timeline-table-details { margin-top: 0.5rem; }
    .timeline-table-details summary { cursor: pointer; color: var(--text-muted); font-size: 0.9rem; }

    .data-table {
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
      margin-top: 0.5rem; border: 1px solid var(--border);
    }
    .data-table th, .data-table td {
      padding: 0.5rem 0.7rem; border: 1px solid var(--border);
      text-align: left; vertical-align: top;
    }
    .data-table th { background: var(--bg-alt); font-weight: 600; }
    .data-table tr:nth-child(even) td { background: var(--bg-alt); }
    .data-table tr.human-row td { background: var(--human-bg); color: var(--human-text); }

    .claims-table .claim-row { cursor: pointer; transition: background 0.15s; }
    .claims-table .claim-row:hover td { background: var(--blue-bg); }
    .claims-table .claim-row:focus-visible { outline: 3px solid var(--blue); outline-offset: -2px; }
    .claim-text { display: block; }
    .claim-detail-inner {
      padding: 0.8rem; background: var(--bg-alt);
      border-radius: 4px; border: 1px dashed var(--border);
    }
    .claim-detail-inner p { margin-bottom: 0.5rem; }

    .evidence-table { margin-top: 0.5rem; }
    .text-preview { max-width: 320px; font-size: 0.83rem; color: var(--text-muted); }
    .note-cell { max-width: 280px; font-size: 0.85rem; }

    .no-evidence-note {
      padding: 0.5rem 0.8rem; background: var(--red-bg);
      border: 1px solid var(--red-border); border-radius: 4px;
      color: var(--red); font-size: 0.88rem; margin-top: 0.5rem;
    }
    .limitation { color: var(--text-muted); font-size: 0.87rem; margin-top: 0.4rem; }

    .reason-codes { padding-left: 1.2rem; }
    .reason-codes li { margin-bottom: 0.2rem; }

    .human-note {
      background: var(--human-bg); border: 1px solid var(--human-border);
      color: var(--human-text); border-radius: 5px;
      padding: 0.5rem 0.8rem; margin: 0.4rem 0;
      display: flex; gap: 0.5rem; align-items: flex-start;
    }
    .human-note-text { flex: 1; }

    .audit-table td { font-size: 0.83rem; }

    .origin-table { max-width: 500px; }

    .instruction { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 0.5rem; }
    .empty-note { color: var(--text-muted); font-style: italic; }

    .page-footer {
      margin-top: 3rem; padding-top: 1.2rem;
      border-top: 1px solid var(--border);
      color: var(--text-muted); font-size: 0.85rem;
    }

    code {
      font-family: "Consolas", "Courier New", monospace;
      font-size: 0.85em;
      background: var(--bg-alt); padding: 0.1em 0.3em;
      border-radius: 3px;
    }
    """


def _build_js() -> str:
    return """
    function showScenario(sid) {
      document.querySelectorAll(".scenario-panel").forEach(function(p) {
        p.hidden = true;
      });
      document.querySelectorAll(".tab-btn").forEach(function(b) {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      var panel = document.getElementById("panel-" + sid);
      if (panel) { panel.hidden = false; }
      var btn = document.getElementById("tab-btn-" + sid);
      if (btn) {
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
      }
    }

    function toggleClaimDetail(claimId) {
      var row = document.getElementById("claim-detail-" + claimId);
      if (!row) return;
      var parentRow = row.previousElementSibling;
      var expanded = !row.hidden;
      row.hidden = expanded;
      if (parentRow) {
        parentRow.setAttribute("aria-expanded", String(!expanded));
      }
    }
    """


def main():
    print()
    print("=" * 72)
    print("CLINEXUS DEMO VIEWER — BUILD INDEX.HTML")
    print("=" * 72)

    data = load_results()
    scenarios = data.get("scenarios", [])
    print(f"[OK] Loaded results.json: {len(scenarios)} scenarios, "
          f"generated {data.get('generated_at','?')}")

    html = build_html(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"[OK] Wrote {OUTPUT_PATH} ({size_kb:.1f} KB)")

    # Sanity checks
    has_http = False
    lines_with_http = []
    for i, line in enumerate(html.splitlines(), 1):
        low = line.lower().strip()
        # Allow http references in text content (evidence metadata etc.)
        # Flag only src=, href= (attribute) type network resource loads
        if ("src=" in low or "href=" in low or "@import" in low) and ("http://" in low or "https://" in low):
            has_http = True
            lines_with_http.append(i)

    if has_http:
        print(f"[WARN] Possible external http reference on lines: {lines_with_http}")
    else:
        print("[OK] No external http resource references found (src=, href=, @import).")

    scenario_ids_in_html = []
    for sid in ["A", "B", "C", "D"]:
        if f'id="panel-{sid}"' in html:
            scenario_ids_in_html.append(sid)
    print(f"[OK] Scenario panels present: {scenario_ids_in_html}")

    if len(scenario_ids_in_html) < 4:
        print("[WARN] Not all 4 scenario panels found in output!")
    else:
        print("[OK] All 4 scenario panels verified in index.html.")

    print(f"\n  To open: double-click {OUTPUT_PATH}")
    print(f"  Or run:  start {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()
