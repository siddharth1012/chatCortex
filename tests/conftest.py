import pytest

from chatcortex import TaskSpecification, build_demo_registry


@pytest.fixture
def registry():
    return build_demo_registry(
        capabilities=("retrieval", "generation", "verification")
    )


@pytest.fixture
def task():
    return TaskSpecification(
        required_capabilities=["retrieval", "generation", "verification"]
    )
