"""
app.py - Streamlit user interface.

This file ONLY builds the interface. The real work happens in:
  utils/file_parser.py   -> text extraction
  utils/ocr_processor.py -> OCR
  agents/*.py            -> the four AI agents
  utils/claude_client.py -> Claude API

Results are kept in st.session_state, so moving between pages or editing the
proposal never repeats an API call (and never costs extra money).
"""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from agents.job_analyzer import analyze_job
from agents.profile_matcher import match_profile
from agents.reviewer import generate_reviewed_proposal
from utils import ocr_processor
from utils.claude_client import ClaudeClient, ClaudeError
from utils.file_parser import (ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, DocumentError,
                               extract_text, validate_upload)

st.set_page_config(page_title="AI Job Description Analyzer & Proposal Generator",
                   page_icon="📄", layout="wide")

PAGES = ["1. Document Upload", "2. Document Extraction", "3. Job Analysis",
         "4. Profile Matching", "5. Proposal Generation"]

STATUS_LABELS = {"matched": "✅ Matched", "partial": "🟡 Partial",
                 "not_mentioned": "❔ Not Mentioned in CV",
                 "confirmed_missing": "❌ Confirmed gap"}


# ====================================================================== state
def init_state() -> None:
    defaults = {
        "jd_file": None, "cv_file": None,          # {"name": ..., "bytes": ...}
        "jd_extraction": None, "cv_extraction": None,
        "jd_text": "", "cv_text": "", "confirmed_hash": None,
        "job_analysis": None, "job_warnings": [],
        "profile_match": None, "match_warnings": [],
        "proposal": None, "proposal_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def texts_hash() -> str:
    joined = st.session_state.jd_text + "\x00" + st.session_state.cv_text
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def clear_results(from_step: str = "analysis") -> None:
    """Forget AI results that depend on text that has changed."""
    if from_step == "analysis":
        st.session_state.job_analysis, st.session_state.job_warnings = None, []
    st.session_state.profile_match, st.session_state.match_warnings = None, []
    st.session_state.proposal, st.session_state.proposal_text = None, ""
    st.session_state.pop("proposal_editor", None)


def go_to(page: str) -> None:
    st.session_state["goto"] = page
    st.rerun()


def get_client() -> ClaudeClient | None:
    """Create the shared Claude client once. Shows a friendly error if it fails."""
    if "claude_client" not in st.session_state:
        try:
            st.session_state["claude_client"] = ClaudeClient()
        except ClaudeError as error:
            st.error(str(error))
            return None
    return st.session_state["claude_client"]


def run_step(spinner_text: str, function):
    """Run an AI step; show friendly errors instead of crashing."""
    client = get_client()
    if client is None:
        return None
    try:
        with st.spinner(spinner_text):
            return function(client)
    except ClaudeError as error:
        st.error(str(error))
    except ValueError as error:
        st.error(str(error))
    except Exception as error:  # Last safety net: never show a raw traceback.
        st.error(f"Unexpected error: {error}")
    return None


def show_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        st.warning(warning)


# ====================================================================== PAGE 1
def page_upload() -> None:
    st.header("Step 1: Upload documents")
    st.caption(f"Supported formats: {', '.join(sorted(e.upper() for e in ALLOWED_EXTENSIONS))} "
               f"(maximum {MAX_FILE_SIZE_MB} MB each). Scanned PDFs and images are read with OCR.")

    left, right = st.columns(2)
    for column, label, key, uploader_key in (
            (left, "Job Description", "jd_file", "jd_uploader"),
            (right, "Candidate CV", "cv_file", "cv_uploader")):
        with column:
            st.subheader(label)
            uploaded = st.file_uploader(
                f"Upload {label}", type=sorted(ALLOWED_EXTENSIONS), key=uploader_key)
            if uploaded is not None:
                data = uploaded.getvalue()
                try:
                    validate_upload(uploaded.name, len(data))
                    st.session_state[key] = {"name": uploaded.name, "bytes": data}
                except DocumentError as error:
                    st.session_state[key] = None
                    st.error(str(error))
            saved = st.session_state[key]
            if saved:
                st.success(f"Ready: {saved['name']} ({len(saved['bytes']) / 1024:.0f} KB)")

    ready = st.session_state.jd_file and st.session_state.cv_file
    if st.button("Extract text from both documents", type="primary", disabled=not ready):
        extract_documents()


def extract_documents() -> None:
    ok = True
    with st.spinner("Reading documents (OCR can take a little while)..."):
        for label, key, result_key, text_key in (
                ("Job Description", "jd_file", "jd_extraction", "jd_text"),
                ("CV", "cv_file", "cv_extraction", "cv_text")):
            saved = st.session_state[key]
            try:
                result = extract_text(saved["name"], saved["bytes"])
                st.session_state[result_key] = result
                st.session_state[text_key] = result.text
                st.session_state.pop(f"{text_key}_editor", None)  # reseed editor
            except DocumentError as error:
                st.session_state[result_key] = None
                st.error(f"{label}: {error}")
                ok = False
            except Exception as error:
                st.session_state[result_key] = None
                st.error(f"{label}: unexpected error while reading the file ({error}).")
                ok = False
    if ok:
        st.session_state.confirmed_hash = None
        clear_results()
        go_to(PAGES[1])


# ====================================================================== PAGE 2
def page_extraction() -> None:
    st.header("Step 2: Verify extracted text")
    if not (st.session_state.jd_extraction and st.session_state.cv_extraction):
        st.info("Upload and extract both documents first (page 1).")
        return

    ok, message = ocr_processor.check_tesseract()
    st.caption(("OCR available: " if ok else "OCR not available: ") + message)
    st.write("Check the text below and fix any mistakes. This exact text is sent to the AI.")

    tabs = st.tabs(["Job Description", "Candidate CV"])
    for tab, label, result_key, text_key in (
            (tabs[0], "Job Description", "jd_extraction", "jd_text"),
            (tabs[1], "CV", "cv_extraction", "cv_text")):
        with tab:
            result = st.session_state[result_key]
            c1, c2, c3 = st.columns(3)
            c1.metric("File", result.filename)
            c2.metric("Extraction method", result.method_summary)
            c3.metric("Pages", len(result.pages))
            show_warnings(result.warnings)

            if len(result.pages) > 1 or result.pages[0].method == "ocr":
                with st.expander("Page-level details"):
                    st.dataframe([{
                        "Page": p.page_number, "Method": p.method, "Characters": p.char_count,
                        "OCR confidence": f"{p.confidence:.0f}" if p.confidence is not None else "-",
                        "Note": p.warning} for p in result.pages], hide_index=True)

            editor_key = f"{text_key}_editor"
            if editor_key not in st.session_state:
                st.session_state[editor_key] = st.session_state[text_key]
            st.text_area(f"{label} text (editable)", key=editor_key, height=400)
            st.session_state[text_key] = st.session_state[editor_key]

    empty = not (st.session_state.jd_text.strip() and st.session_state.cv_text.strip())
    if empty:
        st.error("Both texts must contain something before you can continue.")
    if st.button("Confirm text and continue", type="primary", disabled=empty):
        current = texts_hash()
        if current != st.session_state.confirmed_hash:
            clear_results()  # text changed -> old AI results are out of date
        st.session_state.confirmed_hash = current
        go_to(PAGES[2])


def require_confirmed() -> bool:
    if st.session_state.confirmed_hash is None:
        st.info("Confirm the extracted text first (page 2).")
        return False
    return True


# ====================================================================== PAGE 3
def keyword_table(items) -> None:
    if not items:
        st.caption("None found.")
        return
    st.dataframe([{"Keyword": i.keyword, "Category": i.category, "Evidence (from job description)": i.evidence}
                  for i in items], hide_index=True)


def page_job_analysis() -> None:
    st.header("Step 3: Job description analysis")
    if not require_confirmed():
        return

    if st.session_state.job_analysis is None:
        st.write("Agent 1 will extract explicit requirements and inferred requirements separately.")
        if st.button("Analyze job description", type="primary"):
            outcome = run_step("Analyzing the job description...",
                               lambda c: analyze_job(c, st.session_state.jd_text))
            if outcome:
                st.session_state.job_analysis, st.session_state.job_warnings = outcome
                st.rerun()
        return

    job = st.session_state.job_analysis
    show_warnings(st.session_state.job_warnings)
    st.subheader(job.job_title)

    tab_explicit, tab_inferred, tab_exp, tab_resp = st.tabs(
        ["Explicit keywords", "Inferred requirements",
         "Experience & qualifications", "Responsibilities"])
    with tab_explicit:
        st.caption("Stated directly in the job description.")
        for title, items in (("Technical skills", job.technical_skills),
                             ("Soft skills", job.soft_skills),
                             ("Tools", job.tools),
                             ("Technologies", job.technologies)):
            st.markdown(f"**{title}**")
            keyword_table(items)
    with tab_inferred:
        st.caption("Reasonable implications. These are NOT stated in the job description.")
        if not job.inferred_requirements:
            st.caption("None.")
        for req in job.inferred_requirements:
            with st.expander(req.requirement):
                st.markdown(f"**Reasoning:** {req.reasoning}")
                st.markdown(f"**Supporting responsibility:** {req.supporting_responsibility}")
    with tab_exp:
        st.markdown("**Experience**")
        keyword_table(job.experience)
        st.markdown("**Educational qualifications**")
        keyword_table(job.qualifications)
    with tab_resp:
        for item in job.responsibilities or ["None found."]:
            st.markdown(f"- {item}")

    if st.button("Continue to profile matching", type="primary"):
        go_to(PAGES[3])


# ====================================================================== PAGE 4
def page_matching() -> None:
    st.header("Step 4: Profile matching")
    if not require_confirmed():
        return
    if st.session_state.job_analysis is None:
        st.info("Analyze the job description first (page 3).")
        return

    if st.session_state.profile_match is None:
        st.write("Agent 2 compares your CV with each requirement. "
                 "The score is calculated by the program, not by the AI.")
        if st.button("Match CV with job requirements", type="primary"):
            outcome = run_step("Comparing the CV with the requirements...",
                               lambda c: match_profile(c, st.session_state.job_analysis,
                                                       st.session_state.cv_text))
            if outcome:
                st.session_state.profile_match, st.session_state.match_warnings = outcome
                st.rerun()
        return

    match = st.session_state.profile_match
    show_warnings(st.session_state.match_warnings)

    top, breakdown = st.columns([1, 2])
    with top:
        st.metric("Overall match score", f"{match.match_score:.0f} / 100")
        st.progress(int(match.match_score))
    with breakdown:
        st.markdown("**Score breakdown**")
        for label, value in match.score_breakdown.items():
            st.caption(f"{label}: {value:.0f}%")
            st.progress(int(value))
    with st.expander("How is the score calculated?"):
        for note in match.scoring_notes:
            st.markdown(f"- {note}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matching skills")
        for item in match.matching_skills or ["None verified."]:
            st.markdown(f"- {item}")
        st.subheader("Candidate strengths")
        for item in match.candidate_strengths or ["None listed."]:
            st.markdown(f"- {item}")
        st.subheader("Relevant experience")
        for item in match.relevant_experience or ["None listed."]:
            st.markdown(f"- {item}")
    with c2:
        st.subheader("Missing or unverified requirements")
        st.caption("'Not Mentioned in CV' means unknown. It does not mean the candidate lacks the skill.")
        for item in match.missing_requirements or ["None."]:
            st.markdown(f"- {item}")
        if match.skill_gaps:
            st.subheader("Confirmed skill gaps")
            for item in match.skill_gaps:
                st.markdown(f"- {item}")
        st.subheader("Improvement suggestions")
        for item in match.improvement_suggestions or ["None."]:
            st.markdown(f"- {item}")

    with st.expander("Requirement-by-requirement details"):
        st.dataframe([{"Requirement": a.requirement, "Type": a.group,
                       "Importance": a.importance, "Status": STATUS_LABELS[a.status],
                       "CV evidence": a.cv_evidence, "Note": a.note}
                      for a in match.assessments], hide_index=True)

    if st.button("Continue to proposal", type="primary"):
        go_to(PAGES[4])


# ====================================================================== PAGE 5
def show_copyable(text: str) -> None:
    """st.code has a built-in copy button in its top-right corner."""
    try:
        st.code(text, language=None, wrap_lines=True)
    except TypeError:  # older Streamlit without wrap_lines
        st.code(text, language=None)


def page_proposal() -> None:
    st.header("Step 5: Proposal")
    if not require_confirmed():
        return
    if st.session_state.profile_match is None:
        st.info("Complete profile matching first (page 4).")
        return

    label = "Regenerate proposal" if st.session_state.proposal else "Generate and review proposal"
    if st.button(label, type="primary"):
        client = get_client()
        if client:
            try:
                with st.status("Working on your proposal...", expanded=True) as status:
                    result = generate_reviewed_proposal(
                        client, st.session_state.job_analysis, st.session_state.profile_match,
                        st.session_state.cv_text, st.session_state.jd_text,
                        on_progress=lambda message: st.write(message))
                    status.update(label="Proposal ready", state="complete")
                st.session_state.proposal = result
                st.session_state.proposal_text = result.review.final_proposal
                st.session_state["proposal_editor"] = result.review.final_proposal
            except (ClaudeError, ValueError) as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Unexpected error: {error}")

    result = st.session_state.proposal
    if result is None:
        return

    review = result.review
    show_warnings(result.warnings)
    if review.approval_status == "approved":
        st.success(f"Review status: APPROVED (revisions used: {result.revisions_used})")
    elif review.approval_status == "needs_revision":
        st.warning("Review status: NEEDS REVISION. Please check the feedback and edit below.")
    else:
        st.error("Review status: REJECTED. The proposal contained serious problems.")

    with st.expander("Reviewer feedback", expanded=review.approval_status != "approved"):
        if review.issues:
            st.dataframe([{"Category": i.category, "Severity": i.severity,
                           "Issue": i.description, "Fixed": "yes" if i.fixed else "no"}
                          for i in review.issues], hide_index=True)
        else:
            st.write("No issues found.")
        if review.corrections:
            st.markdown("**Corrections made**")
            for correction in review.corrections:
                st.markdown(f"- {correction}")
    with st.expander("Original draft (before review)"):
        st.text(result.draft)

    st.subheader("Final proposal (editable)")
    if "proposal_editor" not in st.session_state:
        st.session_state["proposal_editor"] = st.session_state.proposal_text
    st.text_area("Edit your proposal", key="proposal_editor", height=450,
                 label_visibility="collapsed")
    st.session_state.proposal_text = st.session_state["proposal_editor"]
    st.caption(f"{len(st.session_state.proposal_text.split())} words. Edits are not re-reviewed by AI.")

    st.download_button("Download as TXT", data=st.session_state.proposal_text.encode("utf-8"),
                       file_name="proposal.txt", mime="text/plain")
    with st.expander("Copy proposal"):
        st.caption("Use the copy icon at the top-right of the box.")
        show_copyable(st.session_state.proposal_text)


# ====================================================================== main
def sidebar() -> str:
    if "goto" in st.session_state:  # set by go_to(); must happen BEFORE the radio is created
        st.session_state["nav"] = st.session_state.pop("goto")
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("Configuration")
        client_ok = bool(st.session_state.get("claude_client"))
        has_key = bool(os.getenv("ANTHROPIC_API_KEY", "").strip()) or client_ok
        st.write(("✅" if has_key else "❌") + " API key configured (hidden)")
        st.write(f"Model: `{os.getenv('CLAUDE_MODEL', 'not set')}`")
        if st.button("Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    return page


def main() -> None:
    load_dotenv()
    init_state()
    st.title("AI Job Description Analyzer & Proposal Generator")
    page = sidebar()
    {PAGES[0]: page_upload, PAGES[1]: page_extraction, PAGES[2]: page_job_analysis,
     PAGES[3]: page_matching, PAGES[4]: page_proposal}[page]()


main()
