import torch

from hepattn.experiments.cld.task import CLDTask
from hepattn.models.loss import cost_fns, loss_fns


def _dummy_outputs_targets(
    batch_size: int = 2,
    num_queries: int = 4,
    num_vtxd_hits: int = 6,
    num_trkr_hits: int = 8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(7)

    outputs = {
        "flow_logit": torch.randn(batch_size, num_queries, 6),
        "flow_vtxd_logit": torch.randn(batch_size, num_queries, num_vtxd_hits),
        "flow_trkr_logit": torch.randn(batch_size, num_queries, num_trkr_hits),
    }

    targets = {
        "particle_valid": torch.rand(batch_size, num_queries) > 0.3,
        "particle_class_idx": torch.randint(0, 6, (batch_size, num_queries)),
        "particle_vtxd_valid": torch.rand(batch_size, num_queries, num_vtxd_hits) > 0.6,
        "particle_trkr_valid": torch.rand(batch_size, num_queries, num_trkr_hits) > 0.6,
        "vtxd_valid": torch.rand(batch_size, num_vtxd_hits) > 0.1,
        "trkr_valid": torch.rand(batch_size, num_trkr_hits) > 0.1,
    }

    return outputs, targets


def test_cld_cost_defaults_to_object_only_without_hit_weights():
    outputs, targets = _dummy_outputs_targets()

    task = CLDTask(
        name="reco",
        dim=16,
        hits_included=["vtxd", "trkr"],
        hit_cost_weights=None,
    )
    costs = task.cost(outputs, targets)

    assert set(costs) == {"object_bce"}


def test_cld_cost_uses_nested_term_weights_including_sihit():
    outputs, targets = _dummy_outputs_targets()

    task = CLDTask(
        name="reco",
        dim=16,
        hits_included=["vtxd", "trkr"],
        hit_cost_weights={
            "trkr": {"mask_dice": 3.0, "mask_bce": 0.25},
            "sihit": {"mask_dice": 2.5},
        },
    )
    costs = task.cost(outputs, targets)

    assert set(costs) == {"object_bce", "trkr_mask_dice", "trkr_mask_bce", "sihit_mask_dice"}

    expected_trkr_bce = 0.25 * cost_fns["mask_bce"](
        outputs["flow_trkr_logit"].detach().to(torch.float32),
        targets["particle_trkr_valid"].to(torch.float32),
        input_pad_mask=targets["trkr_valid"],
    )
    assert torch.allclose(costs["trkr_mask_bce"], expected_trkr_bce)

    expected_sihit_dice = 2.5 * cost_fns["mask_dice"](
        torch.cat([outputs["flow_vtxd_logit"], outputs["flow_trkr_logit"]], dim=-1).detach().to(torch.float32),
        torch.cat([targets["particle_vtxd_valid"], targets["particle_trkr_valid"]], dim=-1).to(torch.float32),
        input_pad_mask=torch.cat([targets["vtxd_valid"], targets["trkr_valid"]], dim=-1),
    )
    assert torch.allclose(costs["sihit_mask_dice"], expected_sihit_dice)


def test_cld_loss_uses_nested_term_weights():
    outputs, targets = _dummy_outputs_targets()

    task = CLDTask(
        name="reco",
        dim=16,
        hits_included=["vtxd", "trkr"],
        loss_object_mask="valid",
        hit_loss_weights={
            "vtxd": {"mask_dice": 1.2, "mask_bce": 0.3},
            "sihit": {"mask_bce": 0.4},
        },
    )
    losses = task.loss(outputs, targets)

    assert set(losses) == {"object_class_ce", "vtxd_mask_dice", "vtxd_mask_bce", "sihit_mask_bce"}

    expected_sihit_bce = 0.4 * loss_fns["mask_bce"](
        torch.cat([outputs["flow_vtxd_logit"], outputs["flow_trkr_logit"]], dim=-1),
        torch.cat(
            [targets["particle_vtxd_valid"], targets["particle_trkr_valid"]],
            dim=-1,
        ).to(outputs["flow_logit"].dtype),
        object_valid_mask=targets["particle_valid"],
        input_pad_mask=torch.cat([targets["vtxd_valid"], targets["trkr_valid"]], dim=-1),
    )
    assert torch.allclose(losses["sihit_mask_bce"], expected_sihit_bce)
