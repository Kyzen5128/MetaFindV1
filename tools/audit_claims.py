"""Every change this session reported as done, checked against the file itself.

[MASTER 2026-08-28] Required to run before every review submission.

Why it exists: a `str.replace` with no `assert` silently did nothing, and the
change was reported as complete, then used to rebut a third party's correct
finding. `git status` said the file was modified -- it was, by OTHER edits in
the same batch. **A file having changed is not the claimed line having changed.**

Not a test: it asserts nothing about behaviour, only that the text a report
claims is present, is present. Behaviour is what pytest is for.

    python tools/audit_claims.py
"""
from __future__ import annotations

import pathlib
import sys

# (file, needle, what was reported)
CLAIMS = [
    ("tools/chain_to_stage1.sh", "RETIRED predecessor", "chain_after_n05 定位更正"),
    ("tools/chain_to_stage1.sh", 'say "annotations $ANN"', "拿掉假的 n08 artifacts present"),
    ("tools/chain_to_stage1.sh", "scene_splits.py:141", "補上第二個 sem_edge_cache 讀者"),
    ("tools/chain_after_n05.sh", "exit 78", "退休守衛"),
    ("metafind/data/semantic_edges_run.py", "LLM_MODEL = Path(LLM_MODEL_PATH).name", "路徑/名稱分離"),
    ("metafind/data/semantic_edges_run.py", "open_clip:ViT-bigG-14", "TEXT_ENCODER 走 tower"),
    ("metafind/data/semantic_edges_run.py", "EDGE_DIM = 1280", "邊寬 512 -> 1280"),
    ("metafind/data/semantic_edges_run.py", "n08 is text-only", "text-only 守衛"),
    ("metafind/data/semantic_edges_run.py", "appendix_shared_msg", "F8 範圍更正"),
    ("metafind/train/stage1.py", "lr=round(lr_now, 8)", "lr 記實際套用值"),
    ("metafind/train/stage1.py", "for m in root.modules()", "逐子模組還原"),
    ("metafind/train/stage1.py", "m.training = was_training", "賦值不用會遞迴的 train()"),
    ("metafind/train/stage1.py", "set(train_uids) & set(dev_val_uids)", "消費端不相交檢查"),
    ("metafind/train/stage1.py", 'args.phase == "dev" and dev_val_uids', "評估的雙重守衛"),
    ("metafind/train/stage1.py", "def best_paths", "smoke 不覆蓋正式 best"),
    ("metafind/train/stage1.py", "pools_sha256", "best 紀錄帶池子身分"),
    ("metafind/train/stage1.py", "tmp_ckpt.replace(ckpt_path)", "best 原子寫入"),
    ("metafind/train/stage1.py", "HARDCODED LITERAL", "num_workers 未進協定的註明"),
    ("metafind/train/stage1.py", "_submodules_with_trainable_params", "checkpoint 收 BN buffer"),
    ("tools/build_lowclip_sheets.py", "load_view_rgb", "接觸表改黑底"),
    ("tools/build_lowclip_sheets.py", "fill    n/a", "NaN 覆蓋率用琥珀色"),
    ("tools/label_duel.py", '"provenance"', "對決產物帶出處"),
    ("tests/test_train_stage1.py", '"data")', "AST 掃描含 metafind/data"),
    ("tests/test_train_stage1.py", "WHAT THIS CHECK DOES NOT ESTABLISH", "斷言訊息列出盲點"),
]


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    missing = []
    for name, needle, what in CLAIMS:
        path = repo / name
        text = path.read_text() if path.exists() else ""
        ok = needle in text
        print(f"{'OK ' if ok else 'GONE'}  {what:34s} {name}")
        if not ok:
            missing.append(f"{what} ({name}: {needle!r})")
    print()
    if missing:
        print(f"{len(missing)} claimed change(s) are NOT in the file:")
        for m in missing:
            print("  " + m)
        return 1
    print(f"all {len(CLAIMS)} claimed changes are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
