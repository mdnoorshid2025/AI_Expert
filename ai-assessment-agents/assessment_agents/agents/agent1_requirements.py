from ..llm import get_llm
from ..schemas import RequirementPlan
from ..state import AssessmentState


def agent1_analyze_requirements(
    state: AssessmentState
):

    requirements = state[
        "raw_requirements"
    ]

    structured_llm = (
        get_llm(temperature=0.0)
        .with_structured_output(
            RequirementPlan,
            method="json_schema"
        )
    )

    prompt = f"""
You are Agent 1, an assessment requirement analyst.

Subject:
{requirements["subject"]}

Learner level:
{requirements["learner_level"]}

Difficulty:
{requirements["difficulty"]}

Assessment goal:
{requirements["goal"]}

Focus areas:
{requirements["focus_areas"]}

Your job:

1. Normalize the subject name.
2. Identify the major subject areas that should be assessed.
3. Respect the user's requested focus areas.
4. Match the assessment to the learner level.
5. Create a clear assessment objective.
6. Do not generate questions.

There will eventually be exactly 10 MCQs,
so keep the coverage realistic for 10 questions.
"""

    result = structured_llm.invoke(
        prompt
    )

    return {
        "requirement_plan":
            result.model_dump()
    }