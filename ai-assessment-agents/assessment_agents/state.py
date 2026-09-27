from typing import Any, TypedDict


class AssessmentState(TypedDict, total=False):
    raw_requirements: dict[str, Any]
    requirement_plan: dict[str, Any]
    quiz: list[dict[str, Any]]
    answer_key: dict[str, str]
    user_answers: dict[str, str]
    score_percent: float
    correct_count: int
    incorrect_count: int
    evaluation: dict[str, Any]
    performance_analysis: dict[str, Any]
    branch_report: dict[str, Any]
    email_subject: str
    email_body: str
    email_status: str