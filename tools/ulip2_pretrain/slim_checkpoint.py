"""Turn an official-loop ULIP checkpoint (full model state_dict, 10.5 GB with the frozen OpenCLIP
inside) into the release-shaped file `ULIPBackbone` loads: point_encoder.* + pc_projection + logit_scale.
The OpenCLIP weights are dropped, not changed: they were frozen (requires_grad=False) for the whole run
and the backbone re-creates them from open_clip's own cache, as it does for the official release.
Usage: python tools/ulip2_pretrain/slim_checkpoint.py <checkpoint_best.pt> <out.pt>
"""
import hashlib, json, sys
from pathlib import Path
import torch
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
keep = {k: v for k, v in sd.items() if k.startswith("point_encoder.") or k in ("pc_projection", "logit_scale")}
dropped = [k for k in sd if k not in keep]
assert all(k.startswith("open_clip_model.") for k in dropped), dropped[:5]
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": keep, "epoch": ck.get("epoch"), "best_acc1": ck.get("best_acc1"),
            "source": str(src), "note": "slimmed by tools/ulip2_pretrain/slim_checkpoint.py; OpenCLIP dropped (frozen)"}, dst)
rec = {"source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
       "epoch": ck.get("epoch"), "best_acc1_holdout_zero_shot": ck.get("best_acc1"),
       "n_tensors_kept": len(keep), "n_open_clip_tensors_dropped": len(dropped),
       "logit_scale": float(keep["logit_scale"]), "out": str(dst),
       "out_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(), "out_size_bytes": dst.stat().st_size}
Path(str(dst) + ".json").write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
