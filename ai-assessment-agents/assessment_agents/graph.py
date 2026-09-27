from langgraph.checkpoint.memory import (
    InMemorySaver
)

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from .state import AssessmentState
from .routing import route_by_score

from .agents.input_nodes import (
    collect_requirements,
    collect_answers
)

from .agents.agent1_requirements import (
    agent1_analyze_requirements
)

from .agents.agent2_quiz import (
    agent2_generate_quiz
)

from .agents.agent3_evaluator import (
    agent3_evaluate
)

from .agents.agent4_weakness import (
    agent4_weakness
)

from .agents.agent5_strength import (
    agent5_strength
)

from .agents.agent6_report import (
    agent6_generate_report
)

from .email_service import (
    send_email_node
)


def build_graph():

    builder = StateGraph(
        AssessmentState
    )

    # -------------------------
    # Register nodes
    # -------------------------

    builder.add_node(
        "collect_requirements",
        collect_requirements
    )

    builder.add_node(
        "agent1_requirements",
        agent1_analyze_requirements
    )

    builder.add_node(
        "agent2_generate_quiz",
        agent2_generate_quiz
    )

    builder.add_node(
        "collect_answers",
        collect_answers
    )

    builder.add_node(
        "agent3_evaluate",
        agent3_evaluate
    )

    builder.add_node(
        "agent4_weakness",
        agent4_weakness
    )

    builder.add_node(
        "agent5_strength",
        agent5_strength
    )

    builder.add_node(
        "agent6_report",
        agent6_generate_report
    )

    builder.add_node(
        "send_email",
        send_email_node
    )

    # -------------------------
    # Sequential flow
    # -------------------------

    builder.add_edge(
        START,
        "collect_requirements"
    )

    builder.add_edge(
        "collect_requirements",
        "agent1_requirements"
    )

    builder.add_edge(
        "agent1_requirements",
        "agent2_generate_quiz"
    )

    builder.add_edge(
        "agent2_generate_quiz",
        "collect_answers"
    )

    builder.add_edge(
        "collect_answers",
        "agent3_evaluate"
    )

    # -------------------------
    # Conditional flow
    # -------------------------

    builder.add_conditional_edges(
        "agent3_evaluate",
        route_by_score
    )

    # -------------------------
    # Merge both branches
    # -------------------------

    builder.add_edge(
        "agent4_weakness",
        "agent6_report"
    )

    builder.add_edge(
        "agent5_strength",
        "agent6_report"
    )

    # -------------------------
    # Final flow
    # -------------------------

    builder.add_edge(
        "agent6_report",
        "send_email"
    )

    builder.add_edge(
        "send_email",
        END
    )

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )