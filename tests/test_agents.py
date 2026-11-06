"""Tests 12-14: agents with a fake Claude (CV matching, missing skills, bad proposals)."""

import pytest

from agents.job_analyzer import analyze_job
from agents.profile_matcher import build_requirements, calculate_score, match_profile
from agents.reviewer import enforce_review_rules, generate_reviewed_proposal
from schemas import (InferredRequirement, JobAnalysis, KeywordItem, ProfileMatch, ProposalReview,
                     RawProfileAnalysis, RequirementAssessment)

JD = ("Senior Python Developer. Must have 3+ years of Python and Django experience. "
      "AWS experience is a plus. You will build REST APIs and mentor junior developers.")
CV = ("Jane Doe. Python developer with 4 years of Python and Django experience at Acme. "
      "Built REST APIs. BSc Computer Science.")


class FakeClient:
    """Returns prepared answers instead of calling Claude."""

    def __init__(self, structured=None, texts=None):
        self.structured = structured or {}
        self.texts = list(texts or [])
        self.reviews = 0

    def generate_structured(self, system, user, schema, max_tokens=0, **kw):
        answer = self.structured[schema]
        if isinstance(answer, list):
            answer = answer.pop(0)
        return answer

    def generate_text(self, system, user, max_tokens=0):
        return self.texts.pop(0)


def kw(keyword, evidence, category="skill"):
    return KeywordItem(keyword=keyword, category=category, evidence=evidence)


def job_analysis():
    return JobAnalysis(
        job_title="Senior Python Developer",
        technical_skills=[kw("Python", "3+ years of Python"), kw("Django", "Django experience")],
        technologies=[kw("AWS", "AWS experience is a plus")],
        responsibilities=["build REST APIs", "mentor junior developers"],
        inferred_requirements=[{"requirement": "Mentoring ability",
                                "reasoning": "Mentoring juniors needs leadership",
                                "supporting_responsibility": "mentor junior developers"}],
    )


# ---- Agent 1 -----------------------------------------------------------
def test_job_analyzer_removes_unsupported_explicit_items():
    fake = job_analysis()
    fake.technical_skills.append(kw("Kubernetes", "must know Kubernetes clusters"))
    fake.inferred_requirements.append(InferredRequirement(
        requirement="Public speaking", reasoning="x",
        supporting_responsibility="give conference talks"))
    analysis, warnings = analyze_job(FakeClient({JobAnalysis: fake}), JD)
    assert [i.keyword for i in analysis.technical_skills] == ["Python", "Django"]
    assert len(analysis.inferred_requirements) == 1
    assert len(warnings) == 2


def test_job_analyzer_rejects_empty_input():
    with pytest.raises(ValueError):
        analyze_job(FakeClient(), "   ")


# ---- Agent 2: 12. CV with matching skills --------------------------------
def raw(assessments):
    return RawProfileAnalysis(
        assessments=assessments, relevant_experience=["4 years Python at Acme"],
        candidate_strengths=["Strong Python"], improvement_suggestions=["Add AWS details"])


def test_cv_with_matching_skills_scores_high():
    job = job_analysis()
    reqs = build_requirements(job)
    assert [r.importance for r in reqs] == ["required", "required", "preferred", "inferred"]
    answer = raw([
        {"id": 1, "status": "matched", "cv_evidence": "4 years of Python and Django experience"},
        {"id": 2, "status": "matched", "cv_evidence": "Python and Django experience at Acme"},
        {"id": 3, "status": "not_mentioned"},
        {"id": 4, "status": "partial", "cv_evidence": "Python developer with 4 years"},
    ])
    match, _ = match_profile(FakeClient({RawProfileAnalysis: answer}), job, CV)
    # (3*1 + 3*1 + 1.5*0 + 1*0.5) / 8.5 = 76.5
    assert match.match_score == 76.5
    assert "AWS (Not Mentioned in CV)" in match.missing_requirements[0]
    assert match.skill_gaps == []  # not mentioned is NOT a confirmed gap
    assert match.score_breakdown["Potential score if unverified items are confirmed"] > 76.5


# ---- Agent 2: 13. CV with missing skills ---------------------------------
def test_cv_with_missing_skills_scores_low_and_separates_gap_types():
    job = job_analysis()
    answer = raw([
        {"id": 1, "status": "not_mentioned"},
        {"id": 2, "status": "confirmed_missing", "cv_evidence": "BSc Computer Science"},
        {"id": 3, "status": "not_mentioned"},
        {"id": 4, "status": "not_mentioned"},
    ])
    match, _ = match_profile(FakeClient({RawProfileAnalysis: answer}), job, CV)
    assert match.match_score == 0
    assert match.skill_gaps == ["Django"]
    assert len(match.missing_requirements) == 3


def test_invented_cv_evidence_is_not_counted():
    job = job_analysis()
    answer = raw([
        {"id": 1, "status": "matched", "cv_evidence": "Led 20 engineers at Google for 10 years"},
        {"id": 2, "status": "matched"},  # matched with no evidence at all
        {"id": 3, "status": "not_mentioned"},
        {"id": 4, "status": "not_mentioned"},
    ])
    match, _ = match_profile(FakeClient({RawProfileAnalysis: answer}), job, CV)
    assert match.match_score == 0
    assert all(a.status == "not_mentioned" for a in match.assessments)
    assert any("could not be verified" in n for n in match.scoring_notes)


def test_score_is_calculated_in_python_not_by_claude():
    a = lambda s, imp: RequirementAssessment(requirement="x", group="g", importance=imp, status=s)
    score, breakdown, _ = calculate_score([a("matched", "required"), a("confirmed_missing", "required")])
    assert score == 50.0 and breakdown["Required requirements"] == 50.0


# ---- Agents 3 and 4: 14. Proposal containing unsupported claims -----------
def review(status, issues=(), final="Dear client, " + "x" * 50):
    return ProposalReview(approval_status=status, issues=list(issues), final_proposal=final)


UNSUPPORTED = {"category": "unsupported_claim", "severity": "critical",
               "description": "Claims 10 years at Google", "fixed": False}


def test_cannot_approve_with_unresolved_unsupported_claim():
    fixed = enforce_review_rules(review("approved", [UNSUPPORTED]))
    assert fixed.approval_status == "needs_revision"


def test_approval_allowed_when_claim_was_fixed():
    result = enforce_review_rules(review("approved", [{**UNSUPPORTED, "fixed": True}]))
    assert result.approval_status == "approved"


def _match():
    return ProfileMatch(match_score=50, assessments=[RequirementAssessment(
        requirement="Python", group="Technical skill", importance="required",
        status="matched", cv_evidence="4 years of Python")])


def test_revision_loop_stops_after_two_revisions():
    bad = review("approved", [UNSUPPORTED])  # forced to needs_revision
    client = FakeClient({ProposalReview: [bad, bad, bad]},
                        texts=["draft", "revision 1", "revision 2"])
    result = generate_reviewed_proposal(client, job_analysis(), _match(), CV, JD)
    assert result.revisions_used == 2
    assert result.review.approval_status == "needs_revision"
    assert any("not approved" in w for w in result.warnings)


def test_revision_loop_stops_when_approved():
    client = FakeClient({ProposalReview: [review("needs_revision"), review("approved")]},
                        texts=["draft", "revision 1"])
    result = generate_reviewed_proposal(client, job_analysis(), _match(), CV, JD)
    assert result.revisions_used == 1
    assert result.review.approval_status == "approved"
