"""End-to-end UI test: upload -> extraction -> analysis -> matching -> proposal.
Uses Streamlit's AppTest and a fake Claude client (no API calls, no cost)."""

import io
from pathlib import Path

from docx import Document
from streamlit.testing.v1 import AppTest

from schemas import ProposalReview, RawProfileAnalysis
from tests.test_agents import CV, JD, FakeClient, job_analysis

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def docx_with(text: str) -> bytes:
    buffer = io.BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buffer)
    return buffer.getvalue()


def test_full_flow():
    fake = FakeClient(
        structured={
            type(job_analysis()): job_analysis(),
            RawProfileAnalysis: RawProfileAnalysis(
                assessments=[
                    {"id": 1, "status": "matched", "cv_evidence": "4 years of Python and Django"},
                    {"id": 2, "status": "matched", "cv_evidence": "Python and Django experience at Acme"},
                    {"id": 3, "status": "not_mentioned"},
                    {"id": 4, "status": "not_mentioned"}],
                relevant_experience=["4 years Python at Acme"],
                candidate_strengths=["Python"], improvement_suggestions=["Add AWS"]),
            ProposalReview: ProposalReview(
                approval_status="approved", final_proposal="Dear client, " + "I can help. " * 10),
        },
        texts=["Draft proposal text " * 5])

    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["claude_client"] = fake
    at.session_state["jd_file"] = {"name": "jd.docx", "bytes": docx_with(JD)}
    at.session_state["cv_file"] = {"name": "cv.docx", "bytes": docx_with(CV)}
    at.run()
    assert not at.exception

    # Page 1 -> extraction (button moves to page 2)
    at.button[0].click().run()
    assert at.session_state["nav"] == "2. Document Extraction"
    assert "Python" in at.session_state["jd_text"]

    # Page 2 -> confirm
    [b for b in at.button if b.label.startswith("Confirm")][0].click().run()
    assert at.session_state["nav"] == "3. Job Analysis"

    # Page 3 -> analyze
    [b for b in at.button if b.label.startswith("Analyze")][0].click().run()
    assert at.session_state["job_analysis"].job_title == "Senior Python Developer"

    # Page 4 -> match
    at.session_state["nav"] = "4. Profile Matching"
    at.run()
    [b for b in at.button if b.label.startswith("Match")][0].click().run()
    assert at.session_state["profile_match"].match_score > 0

    # Page 5 -> proposal
    at.session_state["nav"] = "5. Proposal Generation"
    at.run()
    [b for b in at.button if b.label.startswith("Generate")][0].click().run()
    assert at.session_state["proposal"].review.approval_status == "approved"
    assert at.session_state["proposal_text"].startswith("Dear client")
    assert not at.exception


def test_missing_api_key_shows_friendly_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("utils.claude_client.load_dotenv", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["jd_file"] = {"name": "jd.docx", "bytes": docx_with(JD)}
    at.session_state["cv_file"] = {"name": "cv.docx", "bytes": docx_with(CV)}
    at.run()
    at.button[0].click().run()
    [b for b in at.button if b.label.startswith("Confirm")][0].click().run()
    [b for b in at.button if b.label.startswith("Analyze")][0].click().run()
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)
    assert not at.exception
