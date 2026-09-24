import pytest

from vpo.model import Stage, default_dag, validate_dag


def test_default_dag_topological_order():
    order = validate_dag(default_dag())
    assert order.index("decode") < order.index("preprocess") < order.index("infer")
    assert order.index("infer") < order.index("postprocess")
    assert order.index("infer") < order.index("export_embedding")


@pytest.mark.parametrize(
    "stages, message",
    [
        ((Stage("a", "cpu", 1, ("b",)), Stage("b", "cpu", 1, ("a",))), "cycle"),
        ((Stage("a", "cpu", 1, ("ghost",)),), "unknown stages"),
        ((Stage("a", "cpu", 1), Stage("a", "gpu", 1)), "duplicate"),
        ((Stage("a", "tpu", 1),), "unknown pool"),
    ],
)
def test_invalid_dags_are_rejected(stages, message):
    with pytest.raises(ValueError, match=message):
        validate_dag(stages)
