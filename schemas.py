from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

class _Base(BaseModel):
    """Shared settings for all models."""

    # Remove spaces around strings and ignore unexpected extra fields.
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _drop_none_values(cls, data: Any) -> Any:
        # Claude sometimes returns null for an empty list. Dropping the key
        # lets the field use its default value (an empty list) instead.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


# --------------------------------------------------------------------------
# Agent 1: Job Analyzer
# --------------------------------------------------------------------------
class KeywordItem(_Base):
    """An EXPLICIT requirement: stated directly in the job description."""

    keyword: str = Field(..., min_length=1, max_length=150,
                         description="The skill, tool, technology or requirement.")
    category: str = Field(..., min_length=1, max_length=80,
                          description="Short category, e.g. 'programming language'.")
    evidence: str = Field(..., min_length=1, max_length=1200,
                          description="Short verbatim quote from the job description.")


class InferredRequirement(_Base):
    """An INFERRED requirement: a reasonable implication, NOT stated directly."""

    requirement: str = Field(..., min_length=1, max_length=300)
    reasoning: str = Field(..., min_length=1, max_length=1200,
                           description="Why this requirement is implied.")
    supporting_responsibility: str = Field(
        ..., min_length=1, max_length=1200,
        description="The responsibility from the job description that implies it.")


class JobAnalysis(_Base):
    """Complete structured result of Agent 1."""

    job_title: str = Field(..., min_length=1, max_length=200)
    technical_skills: list[KeywordItem] = Field(default_factory=list)
    soft_skills: list[KeywordItem] = Field(default_factory=list)
    tools: list[KeywordItem] = Field(default_factory=list)
    technologies: list[KeywordItem] = Field(default_factory=list)
    experience: list[KeywordItem] = Field(default_factory=list)
    qualifications: list[KeywordItem] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    inferred_requirements: list[InferredRequirement] = Field(default_factory=list)

    def explicit_groups(self) -> dict[str, list[KeywordItem]]:
        """Return all explicit keyword lists with display labels."""
        return {
            "Technical skill": self.technical_skills,
            "Soft skill": self.soft_skills,
            "Tool": self.tools,
            "Technology": self.technologies,
            "Experience": self.experience,
            "Qualification": self.qualifications,
        }


# --------------------------------------------------------------------------
# Agent 2: Profile Matcher
# --------------------------------------------------------------------------
Importance = Literal["required", "preferred", "inferred"]
MatchStatus = Literal["matched", "partial", "not_mentioned", "confirmed_missing"]


class RawAssessment(_Base):
    """What Claude returns for ONE requirement (it never returns a score)."""

    id: int = Field(..., description="The requirement number from the prompt.")
    status: MatchStatus = Field(
        ..., description="matched | partial | not_mentioned | confirmed_missing")
    cv_evidence: str = Field(
        default="", max_length=1500,
        description="Short verbatim quote from the CV. Empty if not mentioned.")
    note: str = Field(default="", max_length=800)


class RawProfileAnalysis(_Base):
    """Claude's raw output for Agent 2. Python calculates the score afterwards."""

    assessments: list[RawAssessment] = Field(..., min_length=1)
    relevant_experience: list[str] = Field(default_factory=list)
    candidate_strengths: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


class RequirementAssessment(_Base):
    """A requirement together with the verified assessment of the CV."""

    requirement: str
    group: str
    importance: Importance
    status: MatchStatus
    cv_evidence: str = "Not Mentioned in CV"
    note: str = ""


class ProfileMatch(_Base):
    """Final result of Agent 2 (score calculated in Python, not by Claude)."""

    matching_skills: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(
        default_factory=list,
        description="Missing OR unverified: the CV does not mention them.")
    relevant_experience: list[str] = Field(default_factory=list)
    candidate_strengths: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(
        default_factory=list,
        description="CONFIRMED gaps only: the CV clearly shows a mismatch.")
    improvement_suggestions: list[str] = Field(default_factory=list)
    match_score: float = Field(..., ge=0, le=100)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    assessments: list[RequirementAssessment] = Field(default_factory=list)
    scoring_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Agent 4: Proposal Reviewer
# --------------------------------------------------------------------------
class ReviewIssue(_Base):
    category: Literal["grammar", "tone", "relevance", "keywords", "completeness",
                      "unsupported_claim", "accuracy", "other"]
    severity: Literal["minor", "major", "critical"]
    description: str = Field(..., min_length=1, max_length=1500)
    fixed: bool = Field(
        default=False,
        description="True if the issue is already fixed in final_proposal.")


class ProposalReview(_Base):
    approval_status: Literal["approved", "needs_revision", "rejected"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    final_proposal: str = Field(..., min_length=20,
                                description="The full corrected proposal text.")
