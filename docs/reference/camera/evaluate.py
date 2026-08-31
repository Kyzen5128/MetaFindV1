#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py
- 以物件身分 (taxonomy_id-model_id) 進行 3D ⇄ Text 檢索評估
- 修正 nDCG@5 計算（含 IDCG 正規化）
- 在建模前強制 args.evaluate_3d = True 避免評估時覆寫權重
- 量測六種檢索各自的耗時、GPU/記憶體負擔，並輸出六向平均
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from utils.utils import get_dataset
import models.ULIP_models as models
from utils.tokenizer import SimpleTokenizer
import time

# ------------------------------
# 小工具：debug 列印
# ------------------------------
def debug(name, x):
    try:
        shape = tuple(x.shape)
        dtype = getattr(x, "dtype", None)
        device = getattr(x, "device", None)
        print(f"[DEBUG] {name}: shape={shape}, dtype={dtype}, device={device}")
    except Exception:
        print(f"[DEBUG] {name}: {type(x)}")

# ------------------------------
# GPU 監測（pynvml 可選）
# ------------------------------
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

import threading
import time as _time

class GPUStatSampler:
    def __init__(self, device_index=0, interval=0.05):
        self.device_index = device_index
        self.interval = interval
        self._running = False
        self._thr = None
        self.samples_util = []
        self.samples_mem = []
        self._handle = None

    def start(self):
        self.samples_util.clear()
        self.samples_mem.clear()
        self._running = True

        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception:
                self._handle = None

        def _loop():
            while self._running:
                if self._handle is not None:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                        mem  = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                        self.samples_util.append(util.gpu)                   # %
                        self.samples_mem.append(mem.used / (1024**2))        # MB
                    except Exception:
                        pass
                _time.sleep(self.interval)

        self._thr = threading.Thread(target=_loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def summary(self):
        if len(self.samples_util) == 0:
            return {"avg_util": None, "max_util": None, "max_mem_mb": None}
        return {
            "avg_util": sum(self.samples_util) / len(self.samples_util),
            "max_util": max(self.samples_util),
            "max_mem_mb": max(self.samples_mem) if self.samples_mem else None
        }

# ------------------------------
# 身分映射
# ------------------------------
def build_object_mapping(taxonomy_ids, model_ids):
    object_ids = [f"{t}-{m}" for t, m in zip(taxonomy_ids, model_ids)]
    obj2idxs = defaultdict(list)
    for i, oid in enumerate(object_ids):
        obj2idxs[oid].append(i)
    return object_ids, obj2idxs

# ------------------------------
# 主評估
# ------------------------------
@torch.no_grad()
def compute_retrieval_metrics_object_level(model, loader, device,
                                           chunk_size=512, topk_chunk=1024,
                                           k_ndcg=5, verbose=True):

    import torch.nn.functional as F
    from collections import defaultdict

    def summarize_ranks(ranks, name):
        valid = [r for r in ranks if r != float("inf")]
        if len(valid) == 0:
            print(f"{name}: 無有效排名")
            return {
                "mrr": 0.0, "r1": 0.0, "r5": 0.0, "r10": 0.0,
                "mean": 0.0, "median": 0.0, "n_valid": 0, "n_total": len(ranks)
            }
        r = torch.tensor(valid, dtype=torch.float32)
        stats = {
            "mrr": (1.0 / r).mean().item(),
            "r1":  (r <= 1).float().mean().item(),
            "r5":  (r <= 5).float().mean().item(),
            "r10": (r <= 10).float().mean().item(),
            "mean":   r.mean().item(),
            "median": r.median().item(),
            "n_valid": len(valid),
            "n_total": len(ranks)
        }
        if verbose:
            n = len(r)
            n1, n5, n10, n100 = int((r<=1).sum()), int((r<=5).sum()), int((r<=10).sum()), int((r<=100).sum())
            print(f"{name} Valid {len(valid)}/{len(ranks)}  Mean {stats['mean']:.1f}  Median {stats['median']:.1f}")
            print(f"{name} Rank≤1 {n1} ({n1/n*100:.1f}%)  ≤5 {n5} ({n5/n*100:.1f}%)  ≤10 {n10} ({n10/n*100:.1f}%)  ≤100 {n100} ({n100/n*100:.1f}%)")
        return stats

    def compute_best_ranks(query_feats, gallery_feats, object_ids, obj2idxs, name):
        N = query_feats.size(0)
        ranks = []
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            sims = query_feats[i:j].to(device) @ gallery_feats.to(device).t()  # [q, N]
            sorted_idx = torch.argsort(sims, dim=1, descending=True)           # [q, N]
            for local in range(j - i):
                gid = i + local
                oid = object_ids[gid]
                pos_idxs = obj2idxs[oid]
                row = sorted_idx[local]
                best = float("inf")
                for p in pos_idxs:
                    pos = (row == p).nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        rank = int(pos.item()) + 1
                        if rank < best:
                            best = rank
                ranks.append(best)
        if verbose:
            print(f"\n{name} retrieval completed: {len(ranks)} queries")
        return ranks

    def ndcg_at_k_pair(query_feats, gallery_feats, object_ids, obj2idxs, k=5, label=""):
        """依每個 query 的正樣本數計 IDCG@k，計算平均 nDCG@k。"""
        N = query_feats.size(0)
        scores = []
        for i in range(0, N, topk_chunk):
            j = min(i + topk_chunk, N)
            sims = query_feats[i:j].to(device) @ gallery_feats.to(device).t()   # [q, N]
            topk = min(k, sims.size(1))
            topk_idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True).indices
            for row_idx in range(j - i):
                gid = i + row_idx
                oid = object_ids[gid]
                pos = set(obj2idxs[oid])
                if len(pos) == 0:
                    scores.append(0.0)
                    continue
                dcg = 0.0
                for rank_pos in range(topk_idx.size(1)):  # 0-based → 名次 = rank_pos+1
                    idx = int(topk_idx[row_idx, rank_pos])
                    if idx in pos:
                        dcg += 1.0 / np.log2(rank_pos + 2)
                P = min(len(pos), k)
                idcg = sum(1.0 / np.log2(t + 2) for t in range(P))
                scores.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(scores)) if len(scores) > 0 else 0.0

    # ------------------ 抽特徵 ------------------
    model.eval()
    all_pc, all_txt, all_img = [], [], []
    all_tax, all_mdl = [], []

    for batch in tqdm(loader, desc="Extracting features"):
        tax_ids = batch[0]
        mdl_ids = batch[1]
        txt     = batch[2].to(device).long()
        pc      = batch[3].to(device).float()
        img     = batch[4].to(device)

        out = model(pc, txt, img)
        pc_emb   = F.normalize(out["pc_embed"],   dim=-1)
        txt_emb  = F.normalize(out["text_embed"], dim=-1)
        img_emb  = F.normalize(out["image_embed"], dim=-1)

        all_pc.append(pc_emb.cpu())
        all_txt.append(txt_emb.cpu())
        all_img.append(img_emb.cpu())
        all_tax.extend(tax_ids)
        all_mdl.extend(mdl_ids)

    pc_feats  = torch.cat(all_pc,  dim=0)
    txt_feats = torch.cat(all_txt, dim=0)
    img_feats = torch.cat(all_img, dim=0)
    N = pc_feats.size(0)

    print(f"PC  features: {tuple(pc_feats.shape)}")
    print(f"TXT features: {tuple(txt_feats.shape)}")
    print(f"IMG features: {tuple(img_feats.shape)}")
    print(f"Total samples: {N}")

    object_ids = [f"{t}-{m}" for t, m in zip(all_tax, all_mdl)]
    obj2idxs = defaultdict(list)
    for i, oid in enumerate(object_ids):
        obj2idxs[oid].append(i)
    print(f"Unique objects: {len(obj2idxs)}")

    # ------------------ 六種檢索 ------------------
    results = {}
    retrieval_times = []
    retrieval_profiles = []  # 每個檢索的時間/GPU統計

    pairs = [
        ("S2T (PC→Text)",   pc_feats,  txt_feats),
        ("T2S (Text→PC)",   txt_feats, pc_feats),
        ("S2I (PC→Image)",  pc_feats,  img_feats),
        ("I2S (Image→PC)",  img_feats, pc_feats),
        ("T2I (Text→Image)",txt_feats, img_feats),
        ("I2T (Image→Text)",img_feats, txt_feats),
    ]

    # 決定 GPU index（從 device 推導）
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_index = int(device.split(":")[1])
        except Exception:
            gpu_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    else:
        gpu_index = 0

    for name, q, g in pairs:
        # --- 記憶體峰值統計（PyTorch） ---
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # --- 啟動 GPU 利用率取樣 ---
        sampler = GPUStatSampler(device_index=gpu_index, interval=0.05)
        sampler.start()

        # --- 計時開始 ---
        start_time = time.time()

        # 檢索與排名
        ranks = compute_best_ranks(q, g, object_ids, obj2idxs, name)

        # --- 計時結束 ---
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.synchronize(device)
        end_time = time.time()
        sampler.stop()
        sample_sum = sampler.summary()

        # 峰值顯存（PyTorch 視角）
        peak_mem_mb = None
        cur_alloc_mb = None
        if torch.cuda.is_available() and "cuda" in str(device):
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            cur_alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)

        elapsed = end_time - start_time
        retrieval_times.append(elapsed)

        # 指標統計
        stats = summarize_ranks(ranks, name)
        ndcg5 = ndcg_at_k_pair(q, g, object_ids, obj2idxs, k=k_ndcg, label=name)
        stats["ndcg@5"] = ndcg5
        results[name] = stats

        retrieval_profiles.append({
            "name": name,
            "time_sec": elapsed,
            "avg_gpu_util": sample_sum["avg_util"],   # 可能為 None（未安裝 pynvml）
            "max_gpu_util": sample_sum["max_util"],
            "max_mem_mb_sampling": sample_sum["max_mem_mb"],  # NVML 取樣到的最大顯存
            "peak_mem_mb_torch": peak_mem_mb,         # PyTorch 統計的峰值顯存
            "cur_alloc_mb_torch": cur_alloc_mb,       # 檢索結束時仍佔用的顯存
        })

    # ---- 平均單向檢索時間 ----
    avg_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

    # ------------------ 印表：檢索指標 ------------------
    print("\n" + "=" * 66)
    print("        3D • Image • Text  —  Object-Level Retrieval Results")
    print("=" * 66)
    for name in results:
        s = results[name]
        print(f"{name}:  MRR {s['mrr']:.4f}  R@1 {s['r1']:.4f}  R@5 {s['r5']:.4f}  R@10 {s['r10']:.4f}  nDCG@5 {s['ndcg@5']:.4f}")
    print("=" * 66)
    print("Note: positives = 具有相同 taxonomy_id-model_id 的樣本（物件層級）")

    # ------------------ 印表：每向時間 / GPU / 記憶體 ------------------
    print("\n" + "-" * 66)
    print("GPU/Memory Profiles by Retrieval")
    print("-" * 66)
    has_util = any(p["avg_gpu_util"] is not None for p in retrieval_profiles)
    for p in retrieval_profiles:
        line = (f"{p['name']}: time {p['time_sec']:.4f}s, "
                f"torch_peak {0.0 if p['peak_mem_mb_torch'] is None else p['peak_mem_mb_torch']:.1f}MB, "
                f"torch_alloc_end {0.0 if p['cur_alloc_mb_torch'] is None else p['cur_alloc_mb_torch']:.1f}MB")
        if has_util:
            au = p['avg_gpu_util']; mu = p['max_gpu_util']; sm = p['max_mem_mb_sampling']
            line += (f", avg_util {0.0 if au is None else au:.1f}%, "
                     f"max_util {0.0 if mu is None else mu:.1f}%, "
                     f"nvml_peak_mem {0.0 if sm is None else sm:.1f}MB")
        print(line)

    # 匯總六向平均
    def _avg(lst):
        lst = [x for x in lst if x is not None]
        return sum(lst)/len(lst) if lst else None

    avg_util = _avg([p["avg_gpu_util"] for p in retrieval_profiles]) if has_util else None
    max_util = _avg([p["max_gpu_util"] for p in retrieval_profiles]) if has_util else None
    avg_peak_torch = _avg([p["peak_mem_mb_torch"] for p in retrieval_profiles])
    avg_nvml_peak  = _avg([p["max_mem_mb_sampling"] for p in retrieval_profiles]) if has_util else None

    print("-" * 66)
    print("Summary (averaged over 6 retrievals)")
    print(f"Avg time per retrieval     : {avg_time:.4f}s")
    if has_util and avg_util is not None:
        print(f"Avg GPU util (average)     : {avg_util:.1f}%")
    if has_util and max_util is not None:
        print(f"Avg GPU util (max)         : {max_util:.1f}%")
    if avg_peak_torch is not None:
        print(f"Avg peak CUDA memory (MB)  : {avg_peak_torch:.1f}MB")
    if has_util and avg_nvml_peak is not None:
        print(f"Avg NVML peak memory (MB)  : {avg_nvml_peak:.1f}MB")
    print("-" * 66)

    # 維持原本回傳介面（為相容）
    s2t = results["S2T (PC→Text)"]; t2s = results["T2S (Text→PC)"]
    return (
        {"mrr": s2t["mrr"], "r1": s2t["r1"], "r5": s2t["r5"], "r10": s2t["r10"], "mean": s2t["mean"], "median": s2t["median"]},
        {"mrr": t2s["mrr"], "r1": t2s["r1"], "r5": t2s["r5"], "r10": t2s["r10"], "mean": t2s["mean"], "median": t2s["median"]},
        results["S2T (PC→Text)"]["ndcg@5"],
        results["T2S (Text→PC)"]["ndcg@5"],
    )

# ------------------------------
# main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_addr", type=str,
                    default="/home/klooom/cheng/3d_retrival/ULIP/outputs/ULIP2_Instuct_c/checkpoint_last.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=512)

    # dataset / model
    ap.add_argument("--pretrain_dataset_name",   default="shapenet")
    ap.add_argument("--pretrain_dataset_prompt", default="shapenet_64")
    ap.add_argument("--validate_dataset_name",   default="shapenet")
    ap.add_argument("--validate_dataset_prompt", default="shapenet_64")
    ap.add_argument("--npoints", type=int, default=8192)
    ap.add_argument("--use_height", action="store_true")
    ap.add_argument("--model", default="ULIP_PointBERT")

    args = ap.parse_args()

    # 選定裝置
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # 在建模前，強制 evaluate_3d = True
    setattr(args, "evaluate_3d", True)

    # 載入 checkpoint
    print(f"Loading checkpoint: {args.ckpt_addr}")
    ckpt = torch.load(args.ckpt_addr, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # DataParallel 前綴處理
    sd = {k.replace("module.", ""): v for k, v in state.items()}

    # 建立模型（優先用 ckpt 的 args）
    if "args" in ckpt and hasattr(ckpt["args"], "model"):
        mdl_name = ckpt["args"].model
        setattr(ckpt["args"], "evaluate_3d", True)
        mdl = getattr(models, mdl_name)(args=ckpt["args"])
        print(f"Using model from checkpoint: {mdl_name}")
    else:
        mdl_name = args.model
        mdl = getattr(models, mdl_name)(args=args)
        print(f"Using model from args: {mdl_name}")

    # 載入權重
    missing, unexpected = mdl.load_state_dict(sd, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    mdl = mdl.to(device).eval()
    print("Model loaded successfully!")

    # 準備資料集
    tok  = SimpleTokenizer()
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfm  = T.Compose([T.Resize(224), T.ToTensor(), norm])
    ds   = get_dataset(tfm, tok, args, "val")

    # 關閉評估期增強 / depth 影像（若支援）
    if hasattr(ds, "augment"):
        ds.augment = False
        print("Set dataset.augment = False for evaluation.")
    if hasattr(ds, "picked_image_type"):
        try:
            ds.picked_image_type = ['']  # 僅 RGB
            print("Set dataset.picked_image_type = [''] (avoid depth images).")
        except Exception:
            pass

    # DataLoader
    try:
        from data.dataset_3d import customized_collate_fn
        collate_fn = customized_collate_fn
    except Exception:
        collate_fn = None
        print("customized_collate_fn not found, fallback to default collate.")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(ds)}")

    # 計算檢索與六向平均時間/負擔
    compute_retrieval_metrics_object_level(
        mdl, dl, device, chunk_size=args.chunk_size
    )

if __name__ == "__main__":
    main()
