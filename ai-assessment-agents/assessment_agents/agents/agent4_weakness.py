import json

from ..llm import get_llm
from ..schemas import ImprovementReport
from ..state import AssessmentState


def agent4_weakness(
    state: AssessmentState
):

    structured_llm = (
        get_llm(temperature=0.1)
        .with_structured_output(
            ImprovementReport,
            method="json_schema"
        )
    )

    prompt = f"""
You are Agent 4, a weakness analysis expert.

The learner scored below 80%.

Score:
{state["score_percent"]}%

Question evaluation:
{json.dumps(
    state["evaluation"],
    indent=2
)}

Performance analysis:
{json.dumps(
    state["performance_analysis"],
    indent=2
)}

Create a detailed improvement report.

Focus on:

1. Weak subject areas.
2. Incorrect answers.
3. Misunderstood concepts.
4. Correct concepts the learner should understand.
5. Specific improvement actions.
6. Practice recommendations.
"""

    report = structured_llm.invoke(
        prompt
    )

    return {
        "branch_report":
            report.model_dump()
    }