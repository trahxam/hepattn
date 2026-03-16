import torch

from hepattn.experiments.pixel.task import PixelTrackTask


def _make_inputs(batch_size=2, num_queries=4, dim=16):
    return {
        "query_embed": torch.randn(batch_size, num_queries, dim),
    }


def _make_targets(batch_size=2, num_queries=4):
    return {
        "particle_valid": torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
            ],
            dtype=torch.bool,
        )[:batch_size, :num_queries],
        "particle_x": torch.randn(batch_size, num_queries),
        "particle_y": torch.randn(batch_size, num_queries),
    }


def _make_task(enable_regression: bool) -> PixelTrackTask:
    return PixelTrackTask(
        name="track",
        input_object="query",
        output_object="track",
        target_object="particle",
        classification_losses={"object_bce": 1.0},
        classification_costs={"object_bce": 5.0},
        dim=16,
        enable_regression=enable_regression,
        regression_fields=["x", "y"],
        regression_loss_weight=1.0,
        regression_cost_weight=1.0,
        has_intermediate_loss=True,
        has_first_layer_loss=False,
    )


def test_pixel_track_task_without_regression_only_emits_classification_outputs():
    task = _make_task(enable_regression=False)
    outputs = task(_make_inputs())
    preds = task.predict(outputs)

    assert set(outputs) == {"track_logit", "track_class_prob"}
    assert "track_valid" in preds
    assert "track_x" not in preds
    assert not hasattr(task, "classification_task")
    assert not hasattr(task, "regression_task")
    assert task.outputs == ["track_logit", "track_class_prob"]


def test_pixel_track_task_with_regression_emits_combined_outputs_and_predictions():
    task = _make_task(enable_regression=True)
    outputs = task(_make_inputs())
    preds = task.predict(outputs)

    assert {"track_logit", "track_class_prob", "track_mu", "track_u", "track_ubar"}.issubset(outputs)
    assert "track_valid" in preds
    assert "track_x" in preds
    assert "track_y" in preds


def test_pixel_track_task_combines_classification_and_regression_losses_and_costs():
    task = _make_task(enable_regression=True)
    inputs = _make_inputs()
    targets = _make_targets()
    outputs = task(inputs)

    losses = task.loss(outputs, targets)
    costs = task.cost(outputs, targets)

    assert set(losses) == {"object_bce", "nll"}
    assert set(costs) == {"object_bce", "nll"}
    assert costs["object_bce"].shape == (2, 4, 4)
    assert costs["nll"].shape == (2, 4, 4)
