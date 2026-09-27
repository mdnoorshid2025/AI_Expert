import json

from ..llm import get_llm
from ..schemas import ImprovementReport
from ..state import AssessmentState


def agent5_strength(
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
You are Agent 5, an advanced performance analyst.

The learner scored 80% or higher.

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

Create a detailed report.

Identify:

1. Demonstrated strengths.
2. Topics understood well.
3. Remaining weaknesses, if any.
4. Advanced areas worth learning.
5. Practical next steps.
6. Practice recommendations.
"""

    report = structured_llm.invoke(
        prompt
    )

    return {
        "branch_report":
            report.model_dump()
    }