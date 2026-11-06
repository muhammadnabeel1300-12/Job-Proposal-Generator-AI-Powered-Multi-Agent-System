"""
proposal_writer.py - Agent 3: writes (and revises) the job proposal.

The writer only receives VERIFIED facts: requirements the CV really supports,
with quotes. It never sees unverified claims as facts.
"""

from schemas import JobAnalysis, ProfileMatch
from utils.claude_client import ClaudeClient
from utils.prompts import (PROPOSAL_WRITER_SYSTEM, build_revision_prompt,
                           build_writer_prompt)


def format_verified_facts(match: ProfileMatch) -> str:
    """Summarize ONLY what the CV supports (matched/partial + strengths)."""
    lines = ["VERIFIED SKILLS AND EXPERIENCE (with CV evidence):"]
    for a in match.assessments:
        if a.status in ("matched", "partial"):
            lines.append(f'- {a.requirement} [{a.status}]: "{a.cv_evidence}"')
    if match.relevant_experience:
        lines.append("\nRELEVANT EXPERIENCE:")
        lines += [f"- {x}" for x in match.relevant_experience]
    if match.candidate_strengths:
        lines.append("\nSTRENGTHS:")
        lines += [f"- {x}" for x in match.candidate_strengths]
    lines.append("\nDo NOT claim any other skill or experience.")
    return "\n".join(lines)


def write_proposal(client: ClaudeClient, job: JobAnalysis, match: ProfileMatch,
                   cv_text: str, jd_text: str) -> tuple[str, list[str]]:
    """Return (proposal_text, warnings)."""
    prompt, warnings = build_writer_prompt(
        job.model_dump_json(), format_verified_facts(match), cv_text, jd_text)
    text = client.generate_text(PROPOSAL_WRITER_SYSTEM, prompt, max_tokens=2000)
    return text, warnings


def revise_proposal(client: ClaudeClient, previous: str, issues: list[str],
                    job: JobAnalysis, match: ProfileMatch, cv_text: str,
                    jd_text: str) -> tuple[str, list[str]]:
    """Rewrite a proposal to fix issues found by the reviewer."""
    prompt, warnings = build_revision_prompt(
        previous, issues, job.model_dump_json(), format_verified_facts(match),
        cv_text, jd_text)
    text = client.generate_text(PROPOSAL_WRITER_SYSTEM, prompt, max_tokens=2000)
    return text, warnings
