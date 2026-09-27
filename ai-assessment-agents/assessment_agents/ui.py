from uuid import uuid4

import streamlit as st

from langgraph.types import Command

from assessment_agents.graph import build_graph


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Assessment Agent",
    page_icon="🧠",
    layout="centered"
)


# ---------------------------------------------------------
# Helper: extract LangGraph interrupt
# ---------------------------------------------------------

def get_interrupt_request(result):

    interrupts = result.get(
        "__interrupt__",
        ()
    )

    if not interrupts:
        return None

    return interrupts[0].value


# ---------------------------------------------------------
# Initialize assessment session
# ---------------------------------------------------------

def initialize_assessment():

    if "graph" not in st.session_state:

        # Each browser session receives its own graph instance.
        st.session_state.graph = (
            build_graph()
        )

        # Each assessment receives a unique LangGraph thread.
        st.session_state.thread_id = (
            str(uuid4())
        )

        st.session_state.config = {
            "configurable": {
                "thread_id":
                    st.session_state.thread_id
            }
        }

        # Start LangGraph.
        #
        # The graph will immediately stop at the
        # collect_requirements interrupt.
        result = (
            st.session_state.graph.invoke(
                {},
                st.session_state.config
            )
        )

        request = (
            get_interrupt_request(
                result
            )
        )

        if (
            request is None
            or request.get("type")
            != "requirements"
        ):

            raise RuntimeError(
                "Expected requirements interrupt."
            )

        st.session_state.stage = (
            "requirements"
        )


# ---------------------------------------------------------
# Reset assessment
# ---------------------------------------------------------

def reset_assessment():

    keys_to_remove = [
        "graph",
        "thread_id",
        "config",
        "stage",
        "questions",
        "final_state"
    ]

    for key in keys_to_remove:

        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


# ---------------------------------------------------------
# Initialize
# ---------------------------------------------------------

initialize_assessment()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🧠 AI Assessment")

st.caption(
    "Multi-agent assessment powered by "
    "LangGraph + Gemma 2"
)


# =========================================================
# STAGE 1: REQUIREMENTS
# =========================================================

if st.session_state.stage == "requirements":

    st.subheader(
        "1. Assessment Requirements"
    )

    st.write(
        "Enter your assessment requirements. "
        "The AI agents will create a customized "
        "10-question assessment."
    )

    with st.form(
        "requirements_form"
    ):

        name = st.text_input(
            "Name"
        )

        email = st.text_input(
            "Email address"
        )

        subject = st.text_input(
            "Subject",
            placeholder="Example: Java"
        )

        learner_level = st.selectbox(
            "Learner level",
            [
                "beginner",
                "intermediate",
                "advanced"
            ],
            index=1
        )

        difficulty = st.selectbox(
            "Difficulty",
            [
                "easy",
                "medium",
                "hard",
                "mixed"
            ],
            index=3
        )

        goal = st.text_area(
            "Assessment goal",
            placeholder=(
                "Example: Prepare for a "
                "Senior Java developer interview"
            )
        )

        focus_areas_text = (
            st.text_input(
                "Focus areas",
                placeholder=(
                    "collections, concurrency, "
                    "JVM, streams"
                )
            )
        )

        submitted = (
            st.form_submit_button(
                "Generate Assessment",
                type="primary"
            )
        )

    if submitted:

        if not email.strip():

            st.error(
                "Email address is required."
            )

        elif not subject.strip():

            st.error(
                "Subject is required."
            )

        elif not goal.strip():

            st.error(
                "Assessment goal is required."
            )

        else:

            focus_areas = [
                item.strip()
                for item
                in focus_areas_text.split(",")
                if item.strip()
            ]

            requirements = {
                "name":
                    name.strip()
                    or "Learner",

                "email":
                    email.strip(),

                "subject":
                    subject.strip(),

                "learner_level":
                    learner_level,

                "difficulty":
                    difficulty,

                "goal":
                    goal.strip(),

                "focus_areas":
                    focus_areas
            }

            with st.spinner(
                "Agents are preparing "
                "your assessment..."
            ):

                # Resume requirement interrupt.
                #
                # After this:
                # Agent 1 executes
                # Agent 2 executes
                # Graph stops at collect_answers.
                result = (
                    st.session_state.graph.invoke(
                        Command(
                            resume=requirements
                        ),
                        st.session_state.config
                    )
                )

            request = (
                get_interrupt_request(
                    result
                )
            )

            if (
                request is None
                or request.get("type")
                != "quiz"
            ):

                st.error(
                    "Expected quiz interrupt "
                    "but did not receive one."
                )

            else:

                st.session_state.questions = (
                    request["questions"]
                )

                st.session_state.stage = (
                    "quiz"
                )

                st.rerun()


# =========================================================
# STAGE 2: QUIZ
# =========================================================

elif st.session_state.stage == "quiz":

    st.subheader(
        "2. Assessment"
    )

    st.write(
        "Answer all 10 questions and "
        "submit your assessment."
    )

    questions = (
        st.session_state.questions
    )

    answers = {}

    with st.form(
        "quiz_form"
    ):

        for question in questions:

            st.markdown(
                f"### Q{question['id']}. "
                f"{question['question']}"
            )

            options = (
                question["options"]
            )

            answer = st.radio(
                "Select your answer",
                options=[
                    "A",
                    "B",
                    "C",
                    "D"
                ],
                format_func=lambda option,
                current_options=options:
                    (
                        f"{option}. "
                        f"{current_options[option]}"
                    ),
                index=None,
                key=f"question_{question['id']}"
            )

            answers[
                str(question["id"])
            ] = answer

            st.divider()

        submitted = (
            st.form_submit_button(
                "Submit Assessment",
                type="primary"
            )
        )

    if submitted:

        unanswered = [
            question_id
            for question_id, answer
            in answers.items()
            if answer is None
        ]

        if unanswered:

            st.error(
                "Please answer all questions "
                "before submitting."
            )

        else:

            with st.spinner(
                "Evaluating your answers "
                "and generating your report..."
            ):

                # Resume second interrupt.
                #
                # After this:
                # Agent 3 executes
                # Agent 4 OR Agent 5 executes
                # Agent 6 executes
                # Email node executes
                # Graph reaches END.
                final_state = (
                    st.session_state.graph.invoke(
                        Command(
                            resume=answers
                        ),
                        st.session_state.config
                    )
                )

            st.session_state.final_state = (
                final_state
            )

            st.session_state.stage = (
                "complete"
            )

            st.rerun()


# =========================================================
# STAGE 3: RESULTS
# =========================================================

elif st.session_state.stage == "complete":

    state = (
        st.session_state.final_state
    )

    st.subheader(
        "3. Assessment Result"
    )

    # --------------------------------------
    # Score
    # --------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Score",
        f'{state["score_percent"]}%'
    )

    col2.metric(
        "Correct",
        state["correct_count"]
    )

    col3.metric(
        "Incorrect",
        state["incorrect_count"]
    )

    # --------------------------------------
    # Performance analysis
    # --------------------------------------

    st.subheader(
        "Performance Analysis"
    )

    analysis = state[
        "performance_analysis"
    ]

    st.write(
        analysis.get(
            "summary",
            ""
        )
    )

    if analysis.get("strengths"):

        st.markdown(
            "**Strengths**"
        )

        for strength in (
            analysis["strengths"]
        ):

            st.write(
                f"✅ {strength}"
            )

    if analysis.get("weaknesses"):

        st.markdown(
            "**Areas for improvement**"
        )

        for weakness in (
            analysis["weaknesses"]
        ):

            st.write(
                f"📌 {weakness}"
            )

    # --------------------------------------
    # Detailed branch report
    # --------------------------------------

    report = state[
        "branch_report"
    ]

    st.subheader(
        report.get(
            "headline",
            "Improvement Plan"
        )
    )

    if report.get(
        "improvement_actions"
    ):

        st.markdown(
            "**Recommended actions**"
        )

        for action in (
            report[
                "improvement_actions"
            ]
        ):

            st.write(
                f"• {action}"
            )

    if report.get(
        "practice_plan"
    ):

        st.markdown(
            "**Practice plan**"
        )

        for item in (
            report[
                "practice_plan"
            ]
        ):

            st.write(
                f"• {item}"
            )

    # --------------------------------------
    # Email
    # --------------------------------------

    st.subheader(
        "Email Report"
    )

    email_status = (
        state["email_status"]
    )

    if email_status.startswith(
        "Email sent successfully"
    ):

        st.success(
            email_status
        )

    elif email_status.startswith(
        "Email sending is disabled"
    ):

        st.info(
            email_status
        )

    else:

        st.error(
            email_status
        )

    with st.expander(
        "View generated email"
    ):

        st.markdown(
            f'**Subject:** '
            f'{state["email_subject"]}'
        )

        st.text(
            state["email_body"]
        )

    # --------------------------------------
    # New assessment
    # --------------------------------------

    st.divider()

    if st.button(
        "Start New Assessment"
    ):

        reset_assessment()