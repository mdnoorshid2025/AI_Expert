from typing import Literal

from .config import PASS_THRESHOLD
from .state import AssessmentState


def route_by_score(
    state: AssessmentState
) -> Literal[
    "agent4_weakness",
    "agent5_strength"
]:

    if (
        state["score_percent"]
        >= PASS_THRESHOLD
    ):
        return "agent5_strength"

    return "agent4_weakness"