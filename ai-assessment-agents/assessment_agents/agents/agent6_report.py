import json

from ..llm import get_llm
from ..schemas import EmailDraft
from ..state import AssessmentState


def agent6_generate_report(
    state: AssessmentState
):

    requirements = state[
        "raw_requirements"
    ]

    structured_llm = (
        get_llm(temperature=0.1)
        .with_structured_output(
            EmailDraft,
            method="json_schema"
        )
    )

    prompt = f"""
You are Agent 6, the final assessment report agent.

Learner:
{requirements["name"]}

Subject:
{requirements["subject"]}

Score:
{state["score_percent"]}%

Correct answers:
{state["correct_count"]}/10

Overall analysis:
{json.dumps(
    state["performance_analysis"],
    indent=2
)}

Detailed report:
{json.dumps(
    state["branch_report"],
    indent=2
)}

Create a professional assessment email.

Include:

1. Meaningful email subject.
2. Greeting.
3. Assessment score.
4. Overall performance summary.
5. Strength areas.
6. Areas requiring improvement.
7. Recommended next steps.
8. Practice recommendations.
9. Encouraging closing.

Never invent or modify assessment scores.
"""

    draft = structured_llm.invoke(
        prompt
    )

    return {
        "email_subject":
            draft.subject,

        "email_body":
            draft.body
    }