# Review brief -- 2026-09-02, the eleven-view renderer, the ESSGNN recipe, the Stage 2 lambda

REVIEW ONLY. Do not edit, run, commit, or execute anything that touches the GPU
(Blender, the renderer, gemma, training). Unit tests that mock Blender are fine.
Report findings as text. Nothing in this changeset has been executed on the
corpus; the smoke figures quoted below came from a 64-sample Stage 2 smoke.

Repo: /home/kyzen/MetaFindV1 -- MetaFind paper reproduction. Research correctness
and traceability outrank convenience. Comment density is deliberate; ignore style.

Code state under review: commit b296616 (working tree clean). The code-bearing
range is f8df428..7785679; the three commits after it touch only the ledger and
disk cleanup notes.

  git -C /home/kyzen/MetaFindV1 diff f8df428..7785679 -- metafind tests tools

Read the real files, not only the diff. Files under docs/paper/ in that range are
downloaded upstream papers (EGNN v3, ULIP-2 v4) -- sources to read, not code to
review.

This is the common baseline for three reviewers. Codex reviews the CODE (logic,
control flow, foot-guns). The ULIP2 Block Reviewer reviews items 1-3 and F-1/F-2
against the paper and the vendored renderer. The ESSGNN Block Reviewer reviews
items 4-6 against the EGNN paper, the EGNN repo and the ledger's rulings.

## Files under review

1. metafind/data/render_blender.py   -- camera list patched to eleven views, one orbit
2. metafind/data/renders.py          -- RENDERER_VERSION 7, N_VIEWS 11, sidecar camera block
3. metafind/train/stage1.py          -- N_VIEWS_PER_ASSET 11
4. metafind/models/essgnn.py         -- Pool "normalised_sum", MlpStructure "egnn_appendix"
5. metafind/models/resolve_stage2.py -- ARCH_DECISIONS / STAGE2_DECISIONS values
6. metafind/train/stage2.py          -- init_lambda read from stage2_protocol; restore prefixes
7. tests/test_renders.py, tests/test_train_stage1.py
8. tools/probes/stage2_smoke_seven_checks.py, tools/probes/data_scan_against_paper.py (new)

Rulings the changes claim to implement, all in workflow/DECISION_LEDGER.md:
DL-075 (ESSGNN takes EGNN's QM9 values), DL-077 item 8 (pooling = normalised
sum), DL-078 (lambda starts at ten percent of the fused query's norm), DL-079
(strict re-render, one orbit of eleven), DL-080 (eleven views implemented).

## What each change claims. Check the claim, do not guess intent.

### 1. render_blender.py -- eleven views on one orbit

PAPER FACT, 2methdology.tex:28: "rendered from 11 orthogonal viewpoints". WHICH
eleven is not stated; Kyzen chose one orbit of eleven equal azimuths at 20 deg
elevation, perspective, 512 px, black composite (IMPLEMENTATION CHOICE).

1a. Claim: the vendored OpenShape script `metafind/vendor/openshape/render_single_glb.py`
    holds a hardcoded twelve-entry `views` list (line 172), three polar rings of
    four, and `--num_images 11` would take `views[0..10]` (line 185-187): two
    rings plus three of the third. So `_patched_script` replaces the whole
    literal through the same asserted-edit mechanism as the other patches.
    CHECK: the replacement anchor is byte-exact to the vendored block, the
    assert fires on zero or multiple matches, and no OTHER use of `num_images`
    in the vendored script changes behaviour at 11 (a random-view branch, a
    modulo, a hardcoded 12 anywhere else).

1b. Claim: in `sample_camera_loc(phi, theta, r)` (vendored line 149-154) `phi`
    is the polar angle from +Z and `theta` the azimuth, so elevation 20 deg is
    phi = 70 deg, and `_views_literal()` writes `[radians(70), radians(az)]`.
    CHECK the convention against the vendored code, and CHECK what "up" is
    after the glTF import in that script (glTF is +Y up; Blender's importer
    converts to +Z by default). If the script overrides that, the orbit is not
    where the sidecar says it is.

1c. Claim: `VIEW_AZIMUTHS_DEG` rounds to six decimals and the sidecar's camera
    block is read off these constants (`renders.py`, the `camera` record). CHECK
    the recorded camera is the rendered camera: same rounding, same order, same
    unit.

1d. `demo()` now asserts eleven files, RGBA, alpha used, and silhouette areas
    not all equal. The old ring test (three groups of four) is gone. CHECK
    whether anything rendered would still catch a silent revert to upstream's
    three-ring list. `tests/test_renders.py` checks equal spacing on the
    CONSTANTS, not on pixels. Say whether that is sufficient and why.

### 2. renders.py -- version 7, eleven live views

2a. `RENDERER_VERSION = 7`; `is_complete` rejects any other version with `!=`
    (line ~699) so all 46,024 existing sidecars are stale and re-rendered.
    CHECK `rebuild_index` writes only complete (version 7) records, so a
    partially re-rendered corpus cannot leave version 6 rows in
    `renders_index.jsonl`.

2b. The sidecar's camera block carries `camera_layout`, `camera_layout_source`,
    `n_views_source`. CHECK that the text labels the count as the paper's and
    the layout as Kyzen's choice, never the layout as a paper fact.

2c. `implementation_fingerprint()` hashes render_blender.py and the vendored
    script. The patched `views` literal is generated from constants in
    render_blender.py, so it is covered. CHECK that claim.

2d. Stale prose: `render_asset`'s docstring still says "view_00.png ..
    view_11.png"; `process_one`'s comment says "once all twelve exist";
    `renders.py` line 1 says "Render 12 views". Flag every remaining twelve.

### 3. stage1.py -- N_VIEWS_PER_ASSET = 11

Claim: asserted against the loaded `views` matrix in `__getitem__` rather than
trusted. CHECK the assert exists, and grep metafind/ tools/ tests/ for any other
hardcoded 12 that means views.

### 4. essgnn.py -- normalised_sum pooling; EGNN-appendix MLP shapes

4a. DL-077 item 8 (Kyzen): "Pooling = normalised sum (sum over nodes, divided by
    its L2 norm)". Implemented at essgnn.py ~661-671 in two branches
    (single graph, batched): `s = sum; s / (s.norm() + 1e-12)`. CHECK both
    branches implement the same function, the batched norm is per graph
    (`dim=1`), and the empty-graph case cannot NaN.

4b. The reason (recorded, INFERENCE from a measurement): the plain sum made
    lambda * e_layout 27x the fused query at init on the smoke batch
    (DL-076 §五). A unit-norm sum keeps the direction of EGNN's QM9 readout
    (`torch.sum`) at unit scale so lambda alone sets the contribution.
    CHECK that the comment labels this an IMPLEMENTATION CHOICE, not upstream.

4c. `MlpStructure = "egnn_appendix"`: edge MLP Linear-SiLU-Linear-SiLU, node
    MLP Linear-SiLU-Linear, coord MLP Linear-SiLU-Linear(bias=False). Sources:
    docs/paper/egnn_source/EGNN.md (appendix, implementation details) and
    /home/kyzen/upstream/egnn/models/gcl.py. CHECK each MLP's shape,
    activation placement, output bias, and whether the coord MLP's last layer
    uses upstream's xavier_uniform gain 0.001 init. Report what upstream does
    that we do not, and label it.

### 5. resolve_stage2.py -- the recorded decisions

`ARCH_DECISIONS`: n_layers 7, hidden 128, pooling normalised_sum, mlp_structure
egnn_appendix, squared distance. `STAGE2_DECISIONS`: init_lambda 9.0,
query_modality_masking "none".

5a. 7 and 128 are claimed as UPSTREAM FACT from /home/kyzen/upstream/egnn
    (`main_qm9.py` argparse defaults). CHECK the actual defaults there,
    including `attention` and `nf`, and whether MetaFind's own text overrides
    any of them (it cites EGNN for drug design, i.e. QM9).

5b. init_lambda 9.0 = 0.1 x 91.4, where 91.4 is the measured L2 norm of the
    fusion output on the smoke batch (DL-078). This is an IMPLEMENTATION
    CHOICE derived from a measurement on ONE batch of an UNTRAINED model.
    CHECK the label, and say whether tying an initial value to a smoke
    measurement is a reproducibility hazard (the number would move if the
    fusion head's init moves). An alternative is welcome, labelled.

5c. CHECK the written stage2_protocol.json records every one of these values
    and that its identity/hash changes when any of them changes, so an old
    protocol cannot be reused silently.

### 6. stage2.py -- init_lambda from the Stage 2 protocol; restore prefixes

6a. `init_lambda` moved from stage1 hyperparameters to stage2_protocol; the
    trainer refuses a protocol without it (line ~797). CHECK the value reaches
    `QueryTower.layout_weight` (dual_tower.py ~208) and that checkpoint restore
    (`load_stage1_checkpoint` / `load_variant`) cannot overwrite it with a
    Stage 1 value; `freeze_for_stage2` matches `endswith("layout_weight")`.

6b. The restore gate's `new_prefixes` was `("query.layout_encoder",
    "layout_weight")` and is now `("query.layout_encoder", "query.layout_weight")`
    -- the parameter is registered as `query.layout_weight` and the gate
    matched by `startswith`, so Stage 2 could never start. CHECK there is a
    test that would fail if the prefix regressed.

### 7-8. Tests and probes

Code only. `stage2_smoke_seven_checks.py` gained `--optimizer flat`,
`--freeze-mask-tokens no` and a `layout_term_scale` measurement.
`data_scan_against_paper.py` is new and read-only over the corpus.

## Two findings by MASTER, to be verified rather than believed

F-1  Stale twelfth view. `render_asset` moves `view_00..view_10` into an
     `asset_dir` that still holds `view_11.png` from the twelve-view era and
     deletes nothing (render_blender.py ~330-335). Consumers read `view_paths`
     from the sidecar (annotate_run.py:1085, encode_text_image.py:524), so the
     stale file is not consumed -- CHECK that, and grep for anything that
     globs `view_*.png` under renders (tests, tools/verify_renders_against_ulip.py,
     view_io). The ULIP2 Engineer is being asked to delete stale views before
     the move-in; review that change when MASTER relays it.

F-2  Chain order. n06 halts with rc 3 when ANY annotation's `image_identity`
     differs from the render's (encode_text_image.py ~450-489, "Retire first,
     then halt"). After the re-render every one of the 45,692 annotations is
     stale, so n06 cannot run before n05 has re-annotated the whole corpus.
     workflow/DATA_PLAN_PAPER_FIRST.md §二 listed n06 before n05; MASTER is
     correcting it to render -> n05 -> n06. CHECK whether any other node has a
     similar identity gate (splits, the query pack, n07b for ProcTHOR).

## What "done" means for this review

VERDICT PASS requires BLOCKER = 0 and MAJOR = 0. Each finding: file:line, what
the code does, what it should do, the evidence class for that "should", and
the failure it produces. State what you could not verify.
