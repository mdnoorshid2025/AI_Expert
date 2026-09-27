from langgraph.types import interrupt

from ..schemas import RequirementInput
from ..state import AssessmentState


def collect_requirements(
    state: AssessmentState
):

    user_input = interrupt({
        "type": "requirements",
        "message":
            "Provide assessment requirements"
    })

    validated = (
        RequirementInput.model_validate(
            user_input
        )
    )

    return {
        "raw_requirements":
            validated.model_dump(
                mode="json"
            )
    }

def collect_answers(
    state: AssessmentState
):

    public_questions = []

    for question in state["quiz"]:

        public_questions.append({
            "id":
                question["id"],

            "topic":
                question["topic"],

            "question":
                question["question"],

            "options": {
                "A":
                    question["option_a"],
                "B":
                    question["option_b"],
                "C":
                    question["option_c"],
                "D":
                    question["option_d"]
            }
        })

    answers = interrupt({
        "type": "quiz",
        "questions": public_questions
    })

    normalized_answers = {
        str(question_id):
            str(answer).strip().upper()
        for question_id, answer
        in answers.items()
    }

    return {
        "user_answers":
            normalized_answers
    }