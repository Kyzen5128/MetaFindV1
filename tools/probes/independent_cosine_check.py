"""Independent recomputation of Table 1 cosine / rank on CPU, hand-written.

Kyzen 2026-09-04: 「你確定你cos計算方法正確?」  This script does NOT import the
evaluator's scorer.  It runs the two towers on N val assets, computes cosine by
the textbook formula (a.b / (|a||b|)), ranks by counting, and compares each
query's own-asset cosine with what the evaluator wrote in per_query_*.jsonl.
Also prints: cos(query, own gallery) vs best other, and a shuffled-target control.
"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/home/kyzen/MetaFindV1")
from metafind import paths
from metafind.train.stage1 import (Stage1Dataset, collate, split_embeds, build_model,
                                   load_protocols, load_stage1_checkpoint, protocol_n_views)
from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CK = Path("/home/kyzen/metafind/metafind_data_attrs/outputs/checkpoints/pilotP1s_split801010_lr1e-4_20260904")
EV = Path("/home/kyzen/metafind/metafind_data_attrs/outputs/eval/eval_pilotP1s_split801010_lr1e-4_20260904/per_query_C_dev_selection.jsonl")

splits = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
uids = splits["dev_val"][:N]
enc, tr, hp = load_protocols()
backbone = ULIPBackbone(BackboneConfig(device="cpu", train_scope="point_encoder_and_fuser"))
model, loss_fn = build_model(enc, tr, hp)
load_stage1_checkpoint(backbone, model, loss_fn, CK / "stage1_best.pt")
model.eval(); backbone.model.eval()
from metafind.data.observation import Observation, ObservationProtocol
obs = ObservationProtocol(positive_policy="same_uid", query=Observation(image="single_view"), gallery=Observation())
ds = Stage1Dataset(uids, enc["image_aggregation"], observation=obs)
loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False, collate_fn=collate, num_workers=2)
G, Q = [], {c: [] for c in ("text", "pc", "full")}
present = {"text": (1, 0, 0), "pc": (0, 0, 1), "full": (1, 1, 1)}
with torch.no_grad():
    for i, batch in enumerate(loader):
        q_emb, g_emb = split_embeds(batch, backbone, "cpu")
        G.append(model.gallery(g_emb).float().numpy())
        n = g_emb["text"].size(0)
        for c, flags in present.items():
            m = torch.tensor(flags, dtype=torch.bool).expand(n, 3)
            Q[c].append(model.query(q_emb, present=m).float().numpy())
        print(f"  batch {i} done", flush=True)
G = np.concatenate(G).astype(np.float64)

def cos_matrix(A, B):            # textbook: a.b / (|a| |b|), no library helper
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i, j] = float(np.dot(a, b)) / (float(np.sqrt(np.dot(a, a))) * float(np.sqrt(np.dot(b, b))))
    return out

ev = {}
for line in EV.read_text().splitlines():
    r = json.loads(line)
    if r["condition"] in Q and r["query_uid"] in set(uids):
        ev[(r["condition"], r["query_uid"])] = r
rng = np.random.default_rng(0)
for c in Q:
    Qc = np.concatenate(Q[c]).astype(np.float64)
    S = cos_matrix(Qc, G)
    own = np.diag(S)
    rank = np.array([1 + int((S[i] > own[i]).sum()) + int(((S[i] == own[i]).sum()) - 1) for i in range(len(uids))])
    other_best = np.array([np.max(np.delete(S[i], i)) for i in range(len(uids))])
    diffs = [abs(own[k] - ev[(c, u)]["target_score"]) for k, u in enumerate(uids) if (c, u) in ev]
    perm = rng.permutation(len(uids))
    shuffled_r1 = np.mean([1 + int((S[i] > S[i, perm[i]]).sum()) == 1 for i in range(len(uids))])
    print(f"[{c:5s}] N={len(uids)} gallery={len(uids)}  R@1(hand)={np.mean(rank==1)*100:5.1f}  "
          f"cos(own) mean={own.mean():.4f} min={own.min():.4f} | best-other mean={other_best.mean():.4f} | "
          f"|q==g| identical? max cos(own)={own.max():.6f} | vs evaluator target_score: n={len(diffs)} max|diff|={max(diffs) if diffs else float('nan'):.2e} | "
          f"shuffled-target R@1={shuffled_r1*100:.1f}")
