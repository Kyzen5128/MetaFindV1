"""The gallery a sensitivity probe scores against must be the real one.

[KYZEN 2026-09-03] "先把 evaluator parity 修好." Named after the defect it
refutes, so a later edit that reintroduces it fails by name rather than by a
number nobody checks.

THE DEFECT. `tools/probes/exp_query_observation.py` called the gallery tower
with `"pc": torch.zeros_like(g_text)`. A correctly-shaped zero tensor is not an
absent modality: the fusion Transformer took it as a PRESENT third slot, so the
probe scored against `Fusion_G(text, image, 0)`. Paper sec. 2.4, verbatim: "The
gallery encoder is modality-complete and frozen after pretraining." The repo's
own derangement measurement had already shown this gallery is PC-DOMINANT --
deranging gallery pc takes text R@1 from 66.88 to 1.80 -- so the substitution
removed its strongest signal, and every number three probes produced from it is
withdrawn.

The scorer diverged twice more: ties counted FOR the model, and float32 where
`normalize_for_scoring` mandates float64.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[1] / "tools" / "probes"


def test_the_retracted_probe_refuses_to_produce_numbers():
    from tools.probes import exp_query_observation as m

    for fn in ("gallery_vectors", "recall", "main"):
        with pytest.raises(SystemExit) as e:
            getattr(m, fn)()
        assert "ZERO point cloud" in str(e.value)
    # load_tower was never part of the defect and the replacement imports it.
    assert callable(m.load_tower)


def test_no_probe_hands_the_gallery_tower_a_zero_point_cloud():
    """A probe may omit a modality from the QUERY -- that is the experiment --
    but never from the gallery, and never by substituting zeros for absence."""
    offenders = []
    for p in sorted(PROBES.glob("*.py")):
        src = p.read_text()
        for call in re.finditer(r"\.gallery\(\s*\{(.{0,400}?)\}", src, re.S):
            if re.search(r'"pc"\s*:\s*(torch\.)?zeros', call.group(1)):
                offenders.append(p.name)
    assert not offenders, (
        f"{offenders} build a modality-complete gallery from a zero point "
        "cloud; sec. 2.4 requires all three modalities on the gallery side")


def test_the_replacement_scores_with_the_production_evaluator():
    """Not its own recall(). The retracted probe's private scorer is exactly
    how the tie policy and the float64 requirement came apart."""
    src = (PROBES / "exp_text_length.py").read_text()
    tree = ast.parse(src)

    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)
                and n.module == "metafind.eval.retrieval" for a in n.names}
    assert {"recall_at_k", "condition_mask", "normalize_for_scoring"} <= imported

    local = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "recall" not in local, "a probe must not define its own recall()"


# Measured 2026-09-03 on pilot10b_20260903/stage1_best.pt. Both runs printed
# these to one decimal, and `exp_text_length.py --arm full` reproduced
# `run_retrieval.py` digit for digit on each. Recorded so a future harness
# change that breaks parity has something to be compared against.
PARITY = {
    "C_dev_selection": {"n_gallery": 4569,
                        "R@1": (78.4, 95.0, 92.1, 98.8, 99.9, 98.7, 100.0)},
    "D_dev_val_vs_train": {"n_gallery": 36554,
                           "R@1": (58.0, 84.6, 78.8, 96.5, 99.6, 94.1, 100.0)},
}


def test_the_parity_fixture_covers_all_seven_table1_conditions():
    from metafind.eval.retrieval import QUERY_CONDITIONS

    for name, rec in PARITY.items():
        assert len(rec["R@1"]) == len(QUERY_CONDITIONS), name
