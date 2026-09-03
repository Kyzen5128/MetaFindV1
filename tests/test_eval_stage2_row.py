"""The Table 1 'w/ ESSGNN' row: Stage 2 query weights over their own Stage 1 parent.

[PAPER 3.2] "Using the Stage-1 head reproduces the 'w/o ESSGNN'"; the w/ ESSGNN
row evaluates the Stage 2 head "on Objaverse-LVIS (which lacks layout and
disables ESSGNN)". Two refusals are pinned here because each would otherwise
produce a plausible-looking number under the wrong label.
"""
from __future__ import annotations

import json

import pytest

from metafind.eval import run_retrieval as rr


def test_a_stage2_record_from_another_parent_is_refused(tmp_path):
    rec = tmp_path / "stage2_full.json"
    rec.write_text(json.dumps({"uri": str(tmp_path / "x.pt"), "sha256": "0" * 64,
                               "stage1_checkpoint_sha256": "a" * 64,
                               "lambda_init": {"init_lambda": 0.3}}))
    with pytest.raises(SystemExit) as e:
        rr.load_stage2_over_stage1(str(rec), {"sha256": "b" * 64})
    assert "fine-tuned from Stage 1 checkpoint" in str(e.value)


def test_a_record_that_is_not_a_stage2_record_is_refused(tmp_path):
    rec = tmp_path / "stage1_ckpt.json"            # a Stage 1 record by mistake
    rec.write_text(json.dumps({"uri": str(tmp_path / "x.pt"), "sha256": "0" * 64,
                               "stage1_checkpoint_sha256": "b" * 64}))
    with pytest.raises(SystemExit) as e:
        rr.load_stage2_over_stage1(str(rec), {"sha256": "b" * 64})
    assert "lambda_init" in str(e.value)


def test_overlay_refuses_a_state_that_skips_the_query_fusion():
    """Scoring the w/ ESSGNN row with Stage 1 query weights is the failure the
    coverage check exists for."""
    import torch
    from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
    from metafind.models.fusion import FusionConfig

    f = FusionConfig(dim=8, kind="transformer", hidden=16, n_heads=2, n_layers=1)
    model = MetaFindDualTower(DualTowerConfig(dim=8, tower_sharing="shared_backbone_separate_fusion",
                                              query_fusion=f, gallery_fusion=f,
                                              use_layout=False))
    # a state carrying only lambda-shaped junk: no query.fusion keys
    state = {"gallery.fusion.mask_tokens": torch.zeros(3, 8)}
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s2.pt")
        torch.save({"trainable_state": state}, p)
        with pytest.raises(SystemExit) as e:
            rr.overlay_stage2_weights(model, {"uri": p}, "cpu")
    assert "does not cover" in str(e.value)
