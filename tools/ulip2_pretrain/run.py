"""Run ULIP's official main.py unmodified on MetaFind's corpus. See data/metafind_ulip_dataset.py."""
import os, sys, types, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE); sys.path.insert(0, "/home/kyzen/MetaFindV1")
os.environ.setdefault("METAFIND_DATA", "/home/kyzen/metafind/metafind_data_attrs")
os.environ.setdefault("METAFIND_TEXT_TEMPLATE", "attrs_v1")
# main.py imports wandb at module level; it is only used under --wandb.
if "wandb" not in sys.modules:
    import importlib.machinery
    w = types.ModuleType("wandb"); w.init = w.log = w.watch = lambda *a, **k: None
    w.__spec__ = importlib.machinery.ModuleSpec("wandb", None); sys.modules["wandb"] = w
from metafind.compat import ulip_patch     # torch._six / pointnet2_ops / knn_cuda shims
ulip_patch.apply(patch_fps=False)
from models.pointbert import misc          # noqa: E402
misc.fps = ulip_patch.fps                  # same pure-torch FPS the vendor copy runs under (LVIS zero-shot 50.9 vs paper 50.6)
import data.dataset_3d                     # noqa: E402,F401  upstream registry first
import data.metafind_ulip_dataset          # noqa: E402,F401  registers MetaFindObjaverse / MetaFindObjaverseVal
# PAPER over released code (ULIP-2 §3.3 "freeze it during the pre-training"; Kyzen 2026-09-04
# 「CLIP 文字塔、影像塔凍結」): the released builder only calls .eval() and never sets
# requires_grad=False, so main.py's optimizer would take all 2.5B OpenCLIP parameters (OOM at
# batch 16 on 32 GiB, and not the paper's method). Wrap the builder; upstream files untouched.
import models.ULIP_models as _ulip_models   # noqa: E402
_orig_builder = _ulip_models.ULIP2_PointBERT_Colored
def _frozen_clip_builder(args):
    model = _orig_builder(args)
    n = 0
    for p in model.open_clip_model.parameters():
        p.requires_grad = False; n += 1
    # A frozen tower needs no autograd graph at all; wrapping its two encode calls in
    # no_grad frees ViT-bigG's activations layer by layer (batch 64 OOMs otherwise).
    import torch as _t
    oc = model.open_clip_model
    oc.encode_image = _t.no_grad()(oc.encode_image)
    oc.encode_text = _t.no_grad()(oc.encode_text)
    print(f"[run.py] OpenCLIP frozen per paper: {n} tensors requires_grad=False; encode_* under no_grad", flush=True)
    return model
_ulip_models.ULIP2_PointBERT_Colored = _frozen_clip_builder
# Memory only, numerically identical: gradient checkpointing per Point-BERT block while training.
# The released loop has none (ULIP-2 trained on 80 GB A100s); on 32 GiB batch 64 x 10k points OOMs
# inside the transformer. This is the same trick metafind.models.ulip_backbone uses for Stage 1.
from models.pointbert import point_encoder as _pe   # noqa: E402
import torch.utils.checkpoint as _ckpt              # noqa: E402
def _ckpt_forward(self, x, pos):
    for block in self.blocks:
        if self.training and x.requires_grad:
            x = _ckpt.checkpoint(block, x + pos, use_reentrant=False)
        else:
            x = block(x + pos)
    return x
_pe.TransformerEncoder.forward = _ckpt_forward
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, "main.py"), run_name="__main__")
