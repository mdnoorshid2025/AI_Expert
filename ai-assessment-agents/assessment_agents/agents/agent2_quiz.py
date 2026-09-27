from ..llm import get_llm
from ..schemas import Quiz
from ..state import AssessmentState


def agent2_generate_quiz(
    state: AssessmentState
):

    requirements = state[
        "raw_requirements"
    ]

    plan = state[
        "requirement_plan"
    ]

    structured_llm = (
        get_llm(temperature=0.0)
        .with_structured_output(
            Quiz,
            method="json_schema"
        )
    )

    prompt = f"""
You are Agent 2, an expert assessment author.

Subject:
{plan["normalized_subject"]}

Learner level:
{requirements["learner_level"]}

Difficulty:
{requirements["difficulty"]}

Assessment objective:
{plan["assessment_objective"]}

Coverage areas:
{plan["coverage_areas"]}

Create exactly 10 multiple-choice questions for this learner.

For each question:
1. Use a unique question ID from 1 to 10.
2. Provide four distinct options, A through D.
3. Set exactly one correct option.
4. Include a concise explanation of the correct answer.
5. Keep the content aligned with the requested level and difficulty.

Distribute questions across the coverage areas and honor the learner's
focus areas: {requirements["focus_areas"]}.
"""

    quiz = structured_llm.invoke(
        prompt
    )

    questions = []
    for question_id, question in enumerate(
        quiz.questions,
        start=1
    ):
        question_data = question.model_dump()
        question_data["id"] = question_id
        questions.append(question_data)

    return {
        "quiz": questions,
        "answer_key": {
            str(question["id"]):
                question["correct_option"]
            for question in questions
        }
    }