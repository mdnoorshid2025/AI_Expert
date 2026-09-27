from assessment_agents.routing import (
    route_by_score
)


def test_70_goes_to_agent4():

    state = {
        "score_percent": 70
    }

    assert (
        route_by_score(state)
        == "agent4_weakness"
    )


def test_79_goes_to_agent4():

    state = {
        "score_percent": 79
    }

    assert (
        route_by_score(state)
        == "agent4_weakness"
    )


def test_80_goes_to_agent5():

    state = {
        "score_percent": 80
    }

    assert (
        route_by_score(state)
        == "agent5_strength"
    )


def test_90_goes_to_agent5():

    state = {
        "score_percent": 90
    }

    assert (
        route_by_score(state)
        == "agent5_strength"
    )