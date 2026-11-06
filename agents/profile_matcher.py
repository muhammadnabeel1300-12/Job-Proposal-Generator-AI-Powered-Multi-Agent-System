"""
profile_matcher.py - Agent 2: compares the CV with the job requirements.

HOW THE SCORE IS CALCULATED (documented methodology)
----------------------------------------------------
Claude does NOT produce the score. Claude only judges each requirement and
quotes evidence from the CV. Python then calculates:

    requirement weight:   required = 3.0, preferred = 1.5, inferred = 1.0
    status credit:        matched = 1.0, partial = 0.5,
                          not_mentioned = 0.0, confirmed_missing = 0.0

    score = 100 x (sum of weight x credit) / (sum of weights)

- "required"  = explicit requirement (default).
- "preferred" = explicit requirement whose evidence says nice-to-have, plus, etc.
- "inferred"  = requirement inferred from responsibilities (weakest weight).

Evidence is verified: if Claude's CV quote cannot be found in the CV text,
the item is NOT counted and is treated as "not_mentioned".

"not_mentioned" (unknown) is kept separate from "confirmed_missing"
(the CV clearly shows a mismatch). A second number shows the potential score if
all unverified items turned out to be satisfied.
"""

import re
from dataclasses import dataclass

from schemas import (JobAnalysis, ProfileMatch, RawProfileAnalysis,
                     RequirementAssessment)
from utils.claude_client import ClaudeClient
from utils.prompts import PROFILE_MATCHER_SYSTEM, build_profile_matcher_prompt
from utils.text_checks import evidence_in_text, normalize

IMPORTANCE_WEIGHTS = {"required": 3.0, "preferred": 1.5, "inferred": 1.0}
STATUS_CREDIT = {"matched": 1.0, "partial": 0.5,
                 "not_mentioned": 0.0, "confirmed_missing": 0.0}

_PREFERRED_WORDS = re.compile(
    r"\b(preferred|nice[- ]to[- ]have|plus|bonus|desirable|advantage|beneficial|"
    r"optional|ideally|an asset)\b", re.IGNORECASE)

NOT_MENTIONED_TEXT = "Not Mentioned in CV"


@dataclass
class Requirement:
    id: int
    text: str
    group: str
    importance: str
    context: str  # Extra info shown to Claude (JD quote or reasoning)


# ------------------------------------------------------------ requirements
def build_requirements(job: JobAnalysis) -> list[Requirement]:
    """Turn a JobAnalysis into one numbered, de-duplicated requirement list."""
    requirements: list[Requirement] = []
    seen: set[str] = set()

    for group, items in job.explicit_groups().items():
        for item in items:
            key = normalize(item.keyword)
            if key in seen:
                continue
            seen.add(key)
            importance = "preferred" if _PREFERRED_WORDS.search(item.evidence) else "required"
            requirements.append(Requirement(
                len(requirements) + 1, item.keyword, group, importance,
                f'job description says: "{item.evidence}"'))

    for inferred in job.inferred_requirements:
        key = normalize(inferred.requirement)
        if key in seen:
            continue
        seen.add(key)
        requirements.append(Requirement(
            len(requirements) + 1, inferred.requirement, "Inferred", "inferred",
            f"inferred because: {inferred.reasoning}"))
    return requirements


def _requirements_prompt(requirements: list[Requirement]) -> str:
    return "\n".join(f"{r.id}. [{r.group}] {r.text} ({r.context})" for r in requirements)


# ------------------------------------------------------------ scoring
def calculate_score(assessments: list[RequirementAssessment]
                    ) -> tuple[float, dict[str, float], list[str]]:
    """Return (overall_score, breakdown, notes). Pure Python, no AI."""
    if not assessments:
        return 0.0, {}, ["There were no requirements to score."]

    def percent(items, unknown_credit=None) -> float:
        total = sum(IMPORTANCE_WEIGHTS[a.importance] for a in items)
        if total == 0:
            return 0.0
        earned = 0.0
        for a in items:
            credit = STATUS_CREDIT[a.status]
            if unknown_credit is not None and a.status == "not_mentioned":
                credit = unknown_credit
            earned += IMPORTANCE_WEIGHTS[a.importance] * credit
        return round(100 * earned / total, 1)

    score = percent(assessments)
    breakdown: dict[str, float] = {}
    for importance, label in (("required", "Required requirements"),
                              ("preferred", "Preferred requirements"),
                              ("inferred", "Inferred requirements")):
        subset = [a for a in assessments if a.importance == importance]
        if subset:
            breakdown[label] = percent(subset)

    unknown = [a for a in assessments if a.status == "not_mentioned"]
    potential = percent(assessments, unknown_credit=1.0)
    if unknown:
        breakdown["Potential score if unverified items are confirmed"] = potential

    notes = [
        "Score = 100 x sum(weight x credit) / sum(weight). Weights: required 3, "
        "preferred 1.5, inferred 1. Credit: matched 1, partial 0.5, otherwise 0.",
        f"{len(unknown)} requirement(s) are 'Not Mentioned in CV'. They count as 0 "
        "but are unverified, not confirmed gaps.",
    ]
    return score, breakdown, notes


# ------------------------------------------------------------ main function
def match_profile(client: ClaudeClient, job: JobAnalysis, cv_text: str
                  ) -> tuple[ProfileMatch, list[str]]:
    """Return (ProfileMatch, warnings)."""
    if not cv_text or not cv_text.strip():
        raise ValueError("The CV is empty.")

    requirements = build_requirements(job)
    if not requirements:
        raise ValueError("The job analysis contains no requirements to compare.")

    prompt, warnings = build_profile_matcher_prompt(_requirements_prompt(requirements), cv_text)
    raw = client.generate_structured(
        system=PROFILE_MATCHER_SYSTEM, user=prompt, schema=RawProfileAnalysis,
        max_tokens=8000)

    assessments, notes = _verify_assessments(requirements, raw, cv_text)
    score, breakdown, score_notes = calculate_score(assessments)

    match = ProfileMatch(
        matching_skills=[a.requirement if a.status == "matched"
                         else f"{a.requirement} (partial match)"
                         for a in assessments if a.status in ("matched", "partial")],
        missing_requirements=[f"{a.requirement} ({NOT_MENTIONED_TEXT})"
                              for a in assessments if a.status == "not_mentioned"],
        relevant_experience=raw.relevant_experience,
        candidate_strengths=raw.candidate_strengths,
        skill_gaps=[a.requirement for a in assessments if a.status == "confirmed_missing"],
        improvement_suggestions=raw.improvement_suggestions,
        match_score=score,
        score_breakdown=breakdown,
        assessments=assessments,
        scoring_notes=notes + score_notes,
    )
    return match, warnings


def _verify_assessments(requirements: list[Requirement], raw: RawProfileAnalysis,
                        cv_text: str) -> tuple[list[RequirementAssessment], list[str]]:
    """Attach Claude's judgments to requirements and verify every CV quote."""
    by_id = {a.id: a for a in raw.assessments}
    notes: list[str] = []
    results: list[RequirementAssessment] = []

    for req in requirements:
        raw_a = by_id.get(req.id)
        status, evidence, note = "not_mentioned", "", ""

        if raw_a is None:
            note = "Not assessed by the AI; treated as unverified."
        else:
            status, evidence, note = raw_a.status, raw_a.cv_evidence, raw_a.note
            if status != "not_mentioned" and not (
                    evidence and evidence_in_text(evidence, cv_text)):
                notes.append(
                    f"'{req.text}': the quoted CV evidence could not be verified, "
                    "so it was not counted.")
                status, evidence = "not_mentioned", ""
                note = "Claimed evidence was not found in the CV text."

        results.append(RequirementAssessment(
            requirement=req.text, group=req.group, importance=req.importance,
            status=status, cv_evidence=evidence or NOT_MENTIONED_TEXT, note=note))
    return results, notes
