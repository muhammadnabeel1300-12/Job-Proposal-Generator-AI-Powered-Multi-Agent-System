"""
reviewer.py - Agent 4: checks the proposal and runs the revision loop.

Loop (maximum 2 revisions):
    review -> approved?  yes: stop
                         no : ask the writer to revise, then review again
A proposal is NEVER approved while an unresolved "unsupported_claim" issue exists.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from agents.proposal_writer import revise_proposal, write_proposal
from schemas import JobAnalysis, ProfileMatch, ProposalReview
from utils.claude_client import ClaudeClient
from utils.prompts import REVIEWER_SYSTEM, build_reviewer_prompt

MAX_REVISIONS = 2


@dataclass
class ProposalResult:
    draft: str
    review: ProposalReview
    revisions_used: int = 0
    warnings: list[str] = field(default_factory=list)
    history: list[ProposalReview] = field(default_factory=list)


def review_proposal(client: ClaudeClient, proposal: str, job: JobAnalysis,
                    cv_text: str, jd_text: str) -> tuple[ProposalReview, list[str]]:
    """Review one proposal. Returns (ProposalReview, warnings)."""
    prompt, warnings = build_reviewer_prompt(proposal, job.model_dump_json(), cv_text, jd_text)
    review = client.generate_structured(
        system=REVIEWER_SYSTEM, user=prompt, schema=ProposalReview, max_tokens=4000)
    return enforce_review_rules(review), warnings


def enforce_review_rules(review: ProposalReview) -> ProposalReview:
    """Safety net: do not trust an 'approved' status that contradicts the issues."""
    unresolved_claims = [i for i in review.issues
                         if i.category == "unsupported_claim" and not i.fixed]
    if review.approval_status == "approved" and unresolved_claims:
        review = review.model_copy(update={"approval_status": "needs_revision"})
    return review


def generate_reviewed_proposal(
        client: ClaudeClient, job: JobAnalysis, match: ProfileMatch,
        cv_text: str, jd_text: str,
        on_progress: Callable[[str], None] | None = None) -> ProposalResult:
    """Run Agent 3 and Agent 4 together, with up to two revision attempts."""
    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    progress("Writing the proposal...")
    draft, warnings = write_proposal(client, job, match, cv_text, jd_text)
    result = ProposalResult(draft=draft, review=None, warnings=warnings)  # type: ignore[arg-type]

    current = draft
    for attempt in range(MAX_REVISIONS + 1):
        progress("Reviewing the proposal..." if attempt == 0
                 else f"Reviewing revision {attempt}...")
        review, review_warnings = review_proposal(client, current, job, cv_text, jd_text)
        result.review = review
        result.history.append(review)
        result.warnings += review_warnings

        if review.approval_status == "approved":
            break
        if attempt == MAX_REVISIONS:
            result.warnings.append(
                "The proposal was not approved after the maximum number of revisions. "
                "Please read the reviewer feedback and edit it carefully.")
            break

        progress(f"Revising the proposal (attempt {attempt + 1} of {MAX_REVISIONS})...")
        issues = [f"[{i.severity}/{i.category}] {i.description}"
                  for i in review.issues if not i.fixed]
        # Start from the reviewer's improved version so good fixes are kept.
        current, revise_warnings = revise_proposal(
            client, review.final_proposal, issues, job, match, cv_text, jd_text)
        result.revisions_used += 1
        result.warnings += revise_warnings

    # Remove duplicate warnings but keep order.
    result.warnings = list(dict.fromkeys(result.warnings))
    return result
