# For CLI interaction, you can run the main.py script. 
# It will guide you through the process of collecting 
# assessment requirements, generating a quiz, 
# and collecting answers from the user. 
# The script will also display the final score 
# and email details after the assessment is complete.

from uuid import uuid4

from langgraph.types import Command

from .graph import build_graph


def collect_requirements_from_user():

    print(
        "\n=== Assessment Requirements ==="
    )

    name = input(
        "Name: "
    ).strip()

    email = input(
        "Email: "
    ).strip()

    subject = input(
        "Subject: "
    ).strip()

    learner_level = (
        input(
            "Level "
            "[beginner/intermediate/advanced]: "
        )
        .strip()
        .lower()
    )

    difficulty = (
        input(
            "Difficulty "
            "[easy/medium/hard/mixed]: "
        )
        .strip()
        .lower()
    )

    goal = input(
        "Assessment goal: "
    ).strip()

    focus_input = input(
        "Focus areas "
        "(comma separated): "
    ).strip()

    focus_areas = [
        item.strip()
        for item
        in focus_input.split(",")
        if item.strip()
    ]

    return {
        "name": name or "Learner",
        "email": email,
        "subject": subject,
        "learner_level":
            learner_level,
        "difficulty":
            difficulty,
        "goal": goal,
        "focus_areas":
            focus_areas
    }


def collect_quiz_answers(
    questions
):

    answers = {}

    print(
        "\n=== Assessment Started ==="
    )

    for question in questions:

        print()
        print(
            f'Q{question["id"]}. '
            f'{question["question"]}'
        )

        for option, value in (
            question["options"].items()
        ):

            print(
                f"{option}. {value}"
            )

        while True:

            answer = (
                input(
                    "Your answer: "
                )
                .strip()
                .upper()
            )

            if answer in {
                "A",
                "B",
                "C",
                "D"
            }:
                break

            print(
                "Enter A, B, C or D."
            )

        answers[
            str(question["id"])
        ] = answer

    return answers


def main():

    graph = build_graph()

    config = {
        "configurable": {
            "thread_id":
                str(uuid4())
        }
    }

    graph_input = {}

    while True:

        interrupted = False

        for chunk in graph.stream(
            graph_input,
            config
        ):

            if "__interrupt__" not in chunk:
                continue

            interrupt_info = (
                chunk["__interrupt__"][0]
            )

            request = (
                interrupt_info.value
            )

            if (
                request["type"]
                == "requirements"
            ):

                response = (
                    collect_requirements_from_user()
                )

            elif (
                request["type"]
                == "quiz"
            ):

                response = (
                    collect_quiz_answers(
                        request["questions"]
                    )
                )

            else:

                raise ValueError(
                    "Unknown interrupt type"
                )

            graph_input = Command(
                resume=response
            )

            interrupted = True

            break

        if interrupted:
            continue

        break

    final_state = (
        graph.get_state(
            config
        ).values
    )

    print(
        "\n=============================="
    )

    print(
        "ASSESSMENT COMPLETE"
    )

    print(
        "=============================="
    )

    print(
        f'\nScore: '
        f'{final_state["score_percent"]}%'
    )

    print(
        f'Correct: '
        f'{final_state["correct_count"]}/10'
    )

    print(
        "\nEmail Subject:"
    )

    print(
        final_state["email_subject"]
    )

    print(
        "\nEmail Body:"
    )

    print(
        final_state["email_body"]
    )

    print(
        "\nEmail Status:"
    )

    print(
        final_state["email_status"]
    )


if __name__ == "__main__":
    main()