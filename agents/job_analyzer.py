"""
job_analyzer.py - Agent 1: extracts explicit and inferred requirements.

After Claude answers, Python double-checks the result:
- An explicit keyword must really appear in the job description.
- An inferred requirement must point to a responsibility that really exists.
Anything that fails these checks is removed and reported as a warning.
This enforces "never convert an assumption into an explicit requirement".
"""

from schemas import JobAnalysis
from utils.claude_client import ClaudeClient
from utils.prompts import JOB_ANALYZER_SYSTEM, build_job_analyzer_prompt
from utils.text_checks import evidence_in_text, keyword_in_text


def analyze_job(client: ClaudeClient, jd_text: str) -> tuple[JobAnalysis, list[str]]:
    """Return (JobAnalysis, warnings)."""
    if not jd_text or not jd_text.strip():
        raise ValueError("The job description is empty.")

    prompt, warnings = build_job_analyzer_prompt(jd_text)
    analysis = client.generate_structured(
        system=JOB_ANALYZER_SYSTEM, user=prompt, schema=JobAnalysis, max_tokens=6000)

    warnings += _verify(analysis, jd_text)
    if not any(analysis.explicit_groups().values()):
        warnings.append("No explicit requirements were found. Check the extracted text.")
    return analysis, warnings


def _verify(analysis: JobAnalysis, jd_text: str) -> list[str]:
    """Remove unsupported items in place and return warnings about them."""
    warnings: list[str] = []

    for label, items in analysis.explicit_groups().items():
        kept = [i for i in items
                if evidence_in_text(i.evidence, jd_text) or keyword_in_text(i.keyword, jd_text)]
        removed = [i.keyword for i in items if i not in kept]
        if removed:
            warnings.append(
                f"Removed unsupported {label.lower()} item(s) not found in the "
                f"job description: {', '.join(removed)}.")
        items[:] = kept  # list is edited in place

    known_responsibilities = analysis.responsibilities
    valid_inferred = []
    for req in analysis.inferred_requirements:
        source = req.supporting_responsibility
        supported = (evidence_in_text(source, jd_text)
                     or any(evidence_in_text(source, r) or evidence_in_text(r, source)
                            for r in known_responsibilities))
        if supported:
            valid_inferred.append(req)
        else:
            warnings.append(
                f"Removed inferred requirement '{req.requirement}': its supporting "
                "responsibility was not found in the job description.")
    analysis.inferred_requirements = valid_inferred
    return warnings
