import json

from ..llm import get_llm
from ..schemas import PerformanceAnalysis
from ..state import AssessmentState


def agent3_evaluate(
    state: AssessmentState
):

    questions = state["quiz"]
    answers = state["user_answers"]
    answer_key = state["answer_key"]

    correct_count = 0

    question_results = []

    topic_stats = {}

    for question in questions:

        question_id = str(
            question["id"]
        )

        selected = answers.get(
            question_id
        )

        if selected not in {
            "A",
            "B",
            "C",
            "D"
        }:
            raise ValueError(
                f"Invalid or missing answer "
                f"for question {question_id}"
            )

        correct = answer_key[
            question_id
        ]

        is_correct = (
            selected == correct
        )

        if is_correct:
            correct_count += 1

        options = {
            "A": question["option_a"],
            "B": question["option_b"],
            "C": question["option_c"],
            "D": question["option_d"]
        }

        question_results.append({
            "id":
                question["id"],

            "topic":
                question["topic"],

            "question":
                question["question"],

            "selected_option":
                selected,

            "selected_answer":
                options[selected],

            "correct_option":
                correct,

            "correct_answer":
                options[correct],

            "is_correct":
                is_correct,

            "explanation":
                question["explanation"]
        })

        topic = question["topic"]

        if topic not in topic_stats:

            topic_stats[topic] = {
                "correct": 0,
                "total": 0
            }

        topic_stats[
            topic
        ]["total"] += 1

        if is_correct:

            topic_stats[
                topic
            ]["correct"] += 1

    total_questions = len(
        questions
    )

    incorrect_count = (
        total_questions
        - correct_count
    )

    score = round(
        (
            correct_count
            / total_questions
        ) * 100,
        2
    )

    # Add percentage for each topic.
    for stats in topic_stats.values():

        stats["percentage"] = round(
            (
                stats["correct"]
                / stats["total"]
            ) * 100,
            2
        )

    structured_llm = (
        get_llm(temperature=0.0)
        .with_structured_output(
            PerformanceAnalysis,
            method="json_schema"
        )
    )

    prompt = f"""
You are Agent 3, an expert assessment evaluator.

Python has already calculated the results.
Do not recalculate or modify them.

Overall score:
{score}%

Correct answers:
{correct_count}/{total_questions}

Topic statistics:
{json.dumps(topic_stats, indent=2)}

Detailed question results:
{json.dumps(question_results, indent=2)}

Analyse the learner's performance.

Identify:
1. Strengths.
2. Weak areas.
3. Misconceptions.
4. Patterns across incorrect answers.
5. Important observations.

Do not change the supplied score.
"""

    analysis = structured_llm.invoke(
        prompt
    )

    return {
        "score_percent":
            score,

        "correct_count":
            correct_count,

        "incorrect_count":
            incorrect_count,

        "evaluation": {
            "question_results":
                question_results,

            "topic_stats":
                topic_stats
        },

        "performance_analysis":
            analysis.model_dump()
    }