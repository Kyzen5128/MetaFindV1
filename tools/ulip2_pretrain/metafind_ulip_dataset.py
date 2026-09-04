"""MetaFind's Objaverse-LVIS corpus served to ULIP's OFFICIAL main.py, unmodified.

[KYZEN 2026-09-04 23:5x 「現在去做」「8020拆分」] Train ULIP-2's point encoder with
ULIP's own training loop (`upstream/ULIP/main.py`, `ULIP2_PointBERT_Colored`,
P<->I + P<->T contrastive, OpenCLIP ViT-bigG frozen per the paper) on OUR
80% (train, 36,554); select the epoch by ULIP's own zero-shot classification on
OUR 20% (holdout, 9,138). No upstream file is edited: this module registers two
dataset classes into ULIP's registry and a wrapper (`run.py`) hands main.py the
catalog in this directory.

What each training sample is (mirrors `ShapeNet.__getitem__`'s tuple so that
`main.py:294-297` reads pc=inputs[3], texts=inputs[2], image=inputs[4]):
  taxonomy_id  = the asset's LVIS category
  model_id     = uid
  captions     = ONE tokenized sentence, the same attrs_v1 serialization
                 MetaFind Stage 1 encodes (IMPLEMENTATION CHOICE: ULIP-2 used
                 the BLIP-2 caption of the sampled view; we have one GPT-4o
                 annotation per asset, not per view)
  data         = 10,000 x 6 (pc_norm(xyz) ++ rgb), exactly ULIP's
                 `Objaverse_Lvis_Colored` preprocessing; no augmentation (the
                 repo's augmentation path is xyz-only and breaks on 6 channels)
  image        = ONE of the 12 Blender views, chosen at random each step
                 (ULIP-2 §3.3 "randomly sample its 2D rendered image"),
                 composited on the same black background n06 used, then
                 main.py's train_transform (RandomResizedCrop 224 + ImageNet
                 normalise -- the released main.py's choice, recorded)
"""
import json, random, sys
import numpy as np, torch, torch.utils.data as data
from utils.build import DATASETS
from data.dataset_3d import pil_loader  # noqa: F401  (keeps the upstream import graph identical)

sys.path.insert(0, "/home/kyzen/MetaFindV1")
from metafind.data.view_io import load_view_rgb           # black-background composite, as n06
from metafind.models.resolve_stage1 import serialize_annotation  # attrs_v1 sentence, as Stage 1


def _pc_norm(pc):
    """ULIP `Objaverse_Lvis_Colored.pc_norm`, verbatim."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m


def _uids(config):
    splits = json.loads(open(config.SPLITS_PATH).read())["object"]
    uids = list(splits[config.SPLIT_KEY])
    import os
    limit = int(os.environ.get("MF_ULIP_LIMIT") or config.get("LIMIT", 0) or 0)   # smoke runs only
    if limit:
        uids = uids[:limit]
    return uids


def _load_pc(root, uid):
    z = np.load(f"{root}/pointclouds/{uid}.npz")
    xyz = _pc_norm(z["xyz"].astype(np.float32))
    return torch.from_numpy(np.concatenate([xyz, z["rgb"].astype(np.float32)], axis=1)).float().contiguous()


@DATASETS.register_module()
class MetaFindObjaverse(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_ROOT
        self.uids = _uids(config)
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.npoints = int(config.npoints)
        self.n_views = 12
        self.cat = {}
        self.text = {}
        for u in self.uids:
            a = json.loads(open(f"{self.root}/annotations/{u}.json").read())
            self.cat[u] = a["lvis_category"]
            self.text[u] = serialize_annotation(a)
        print(f"[MetaFindObjaverse] {config.SPLIT_KEY}: {len(self.uids):,} assets, "
              f"{len(set(self.cat.values())):,} LVIS categories, npoints={self.npoints}", flush=True)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        pc = _load_pc(self.root, uid)
        if self.npoints < pc.shape[0]:
            pc = pc[torch.randperm(pc.shape[0])[: self.npoints]]
        tokens = torch.stack([self.tokenizer(self.text[uid])])       # (1, 77) like ShapeNet's one caption
        v = random.randrange(self.n_views)
        image = self.train_transform(load_view_rgb(f"{self.root}/renders/{uid}/view_{v:02d}.png"))
        return self.cat[uid], uid, tokens, pc, image

    def __len__(self):
        return len(self.uids)


@DATASETS.register_module()
class MetaFindObjaverseVal(data.Dataset):
    """ULIP-style zero-shot classification pool: (pc, label, name); `lvis_metadata['all_keys']`
    lists every LVIS category of the full corpus (train+val+test), as ULIP's LVIS eval used
    all 1,156 names, so the candidate set does not shrink to the holdout's categories."""
    def __init__(self, config):
        self.root = config.DATA_ROOT
        self.uids = _uids(config)
        splits = json.loads(open(config.SPLITS_PATH).read())["object"]
        corpus = list(splits["train"]) + list(splits["val"]) + list(splits["test"])
        cats = {}
        for u in corpus:
            cats[u] = json.loads(open(f"{self.root}/annotations/{u}.json").read())["lvis_category"]
        all_keys = sorted(set(cats.values()))
        self.lvis_metadata = {"all_keys": all_keys,
                              "key_to_id": {k: i for i, k in enumerate(all_keys)},
                              "value_to_key_mapping": {u: cats[u] for u in self.uids}}
        print(f"[MetaFindObjaverseVal] {config.SPLIT_KEY}: {len(self.uids):,} assets; "
              f"{len(all_keys):,} candidate categories", flush=True)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        name = self.lvis_metadata["value_to_key_mapping"][uid]
        return _load_pc(self.root, uid), self.lvis_metadata["key_to_id"][name], name

    def __len__(self):
        return len(self.uids)
