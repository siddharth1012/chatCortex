import pytest

from chatcortex import TaskSpecification


def test_valid_task_passes():
    TaskSpecification(required_capabilities=["retrieval"]).validate()


def test_empty_capabilities_raises():
    with pytest.raises(ValueError):
        TaskSpecification(required_capabilities=[]).validate()


def test_invalid_objective_weight_raises():
    task = TaskSpecification(
        required_capabilities=["retrieval"],
        objective_weights={"cost": 1.0, "bogus": 2.0},
    )
    with pytest.raises(ValueError):
        task.validate()
