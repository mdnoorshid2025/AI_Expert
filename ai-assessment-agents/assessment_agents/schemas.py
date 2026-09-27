from typing import Literal

from pydantic import (
    BaseModel,
    EmailStr,
    Field
)


class RequirementInput(BaseModel):
    name: str = "Learner"

    email: EmailStr

    subject: str

    learner_level: Literal[
        "beginner",
        "intermediate",
        "advanced"
    ]

    difficulty: Literal[
        "easy",
        "medium",
        "hard",
        "mixed"
    ]

    goal: str

    focus_areas: list[str] = Field(
        default_factory=list
    )


class RequirementPlan(BaseModel):
    normalized_subject: str

    coverage_areas: list[str]

    assessment_objective: str


class MCQ(BaseModel):
    id: int

    topic: str

    question: str

    option_a: str
    option_b: str
    option_c: str
    option_d: str

    correct_option: Literal[
        "A",
        "B",
        "C",
        "D"
    ]

    explanation: str


class Quiz(BaseModel):
    questions: list[MCQ] = Field(
        min_length=10,
        max_length=10
    )


class PerformanceAnalysis(BaseModel):
    summary: str

    strengths: list[str]

    weaknesses: list[str]

    observations: list[str]


class ImprovementReport(BaseModel):
    headline: str

    strengths: list[str]

    weaknesses: list[str]

    improvement_actions: list[str]

    practice_plan: list[str]


class EmailDraft(BaseModel):
    subject: str

    body: str