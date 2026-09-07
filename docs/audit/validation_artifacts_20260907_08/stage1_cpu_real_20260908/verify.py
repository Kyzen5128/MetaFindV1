"""Reload genuine pretrained components and check actual saved updates."""
import hashlib
import json
import os
import time
from pathlib import Path
from run import ENV, HERE

os.environ.update(ENV)
from metafind.train import stage1
from metafind.train import gallery_index
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone
import torch


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    started = time.monotonic()
    rp = HERE / "data/outputs/checkpoints/actual_cpu_step/stage1_best_ckpt.json"
    record = gallery_index.load_checkpoint_record(rp)
    bound = stage1.load_stage1_model_config(record["uri"], record)
    enc, tr, hp = stage1.effective_stage1_model_inputs(bound, *stage1.load_protocols())
    stage1.seed_training(record["seed"])
    backbone = ULIPBackbone(BackboneConfig(device="cpu", train_scope=tr["train_scope"],
                                           **stage1.stage1_backbone_kwargs(bound)))
    model, loss_fn = stage1.build_model(enc, tr, hp)
    initial_pc = {n: digest_tensor(p) for n, p in backbone.named_trainable_parameters()}
    initial_fuser = {n: digest_tensor(p) for n, p in model.named_parameters() if p.requires_grad}
    frozen_before = {n: digest_tensor(p) for n, p in backbone.model.named_parameters() if not p.requires_grad}
    ckpt = stage1.load_stage1_checkpoint(backbone, model, loss_fn, Path(record["uri"]))
    changed_pc = [n for n, p in backbone.named_trainable_parameters() if digest_tensor(p) != initial_pc[n]]
    changed_fuser = [n for n, p in model.named_parameters() if p.requires_grad and digest_tensor(p) != initial_fuser[n]]
    frozen_after = {n: digest_tensor(p) for n, p in backbone.model.named_parameters() if not p.requires_grad}
    assert changed_pc and any(n.startswith("point_encoder.") for n in changed_pc)
    assert "pc_projection" in changed_pc
    assert any(n.startswith("query.") for n in changed_fuser)
    assert any(n.startswith("gallery.") for n in changed_fuser)
    assert frozen_before == frozen_after
    for section, module in (("backbone_trainable_state", backbone.model),
                            ("tower_trainable_state", model), ("loss_trainable_state", loss_fn)):
        loaded = module.state_dict()
        for name, value in ckpt[section].items():
            assert torch.equal(loaded[name].cpu(), value), (section, name)
            assert torch.isfinite(value).all(), (section, name)
    backbone.model.eval()
    model.eval()
    loss_fn.eval()
    uids = json.loads((HERE / "preparation.json").read_text())["selection_uids"]
    dataset = stage1.Stage1Dataset(uids, enc["image_aggregation"], preload=True)
    batch = stage1.collate([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        q_embeds, g_embeds = stage1.split_embeds(batch, backbone, "cpu")
        q, g = model.query(q_embeds), model.gallery(g_embeds)
        loss = loss_fn(q.float(), g.float())
        assert torch.isfinite(q).all() and torch.isfinite(g).all()
        assert torch.isfinite(loss["loss"]) and loss["temperature"].item() == .5
    prep = json.loads((HERE / "preparation.json").read_text())
    unchanged = {}
    for path, expected in prep["source_sha256"].items():
        with open(path, "rb") as stream:
            unchanged[path] = hashlib.file_digest(stream, "sha256").hexdigest() == expected
    assert all(unchanged.values())
    result = {
        "classification": "Actual fresh pretrained initialization, strict Stage 1 reload and finite CPU forward after one genuine trainer step. Not quality evaluation.",
        "checkpoint_sha256": record["sha256"], "changed_point_parameter_names": changed_pc,
        "changed_fuser_parameter_names": changed_fuser,
        "frozen_parameters_equal_across_restore": len(frozen_before),
        "frozen_scope_limitation": "Checks that restoring a trainable-only checkpoint does not modify the pretrained frozen tensors. Does not record their bytes during the separate training process.",
        "all_saved_tensors_exactly_restored_and_finite": True,
        "query_shape": list(q.shape), "gallery_shape": list(g.shape),
        "restored_forward_loss": float(loss["loss"]), "effective_temperature": float(loss["temperature"]),
        "learnable_loss_parameters": [n for n,p in loss_fn.named_parameters() if p.requires_grad],
        "source_corpus_files_unchanged": unchanged, "elapsed_seconds": time.monotonic() - started,
        "fusion_initialization_method": "Same seed and actual backbone construction before build_model, matching trainer allocation order; compares reconstructed initial heads with saved heads.",
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_saved_tensors_exactly_restored_and_finite", "effective_temperature", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
