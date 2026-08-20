# 支線任務索引

> Master 維護。這是**支線派工索引**，不取代 `主線.md`（總控）、`支線任務.md`（支線設計）、`TASKS.md`（待修項目清單）。
> 建立於 2026-08-20。

## Status 詞彙

`PLANNED` `READY` `ACTIVE` `BLOCKED` `REVIEW` `DONE` `REJECTED`

只有 Master 可以改動本檔與任何 `TASK.md`。支線只寫自己的 `HANDOFF.md` / `CODEX_REVIEW.md`。

---

## 索引

| Branch | Task ID | Task | Status | Depends On | Master Integration |
|---|---|---|---|---|---|
| A | M1 / T4.3 | n06 全量重編碼（含 n05b 前置） | `READY` | — | 待 |
| B | U-16 / T5.3 / M2.3 | 兩塔共享：矛盾登記與證據卷宗（不做決定） | `READY` | — | 待 |
| C | M2.1 / T5.1 | 跑 n09_build_splits，產出三個協定檔 | `BLOCKED` | A、B、D-α、D-γ | 待 |
| D | M3 / T5 | Stage 1 訓練（smoke → 全量） | `BLOCKED` | A、C | 待 |
| E | S2 / T1 | ESSGNN 架構軸重構（**任務定義已由 Master 修正**） | `BLOCKED` | B（檔案衝突）、D-δ | 待 |
| F | S3 / T2+T3 | n08 節點文字補真值 + n07 支撐方向 | `BLOCKED` | A（GPU）、研究決定 | 待 |
| G | M4 | gallery 索引 n11 → G4 → n12 | `PLANNED` | D | — |
| H | M5 / n13 | Stage 2 訓練 | `PLANNED` | G、E、F | — |
| I | M6 / n15 | **實作** n15_eval_retrieval（目前零程式碼） | `PLANNED` | G | — |
| J | S4 / T6 | Table 2 評估管線（n15a/b/c、n16、n17） | `PLANNED` | U-27、H、I | — |

**G–J 尚未建立資料夾。** 它們的 Scope 取決於尚未做出的決定與尚未存在的產出，
現在寫 Task Card 等於捏造 —— 依 `.claude/rules/research-rigor.md` §2，不做。

---

## 平行性與檔案衝突

```
Master
├── A  n06 重編碼      寫 data/outputs/          ← 與 B 無衝突
├── B  U-16 證據登記    寫 docs/audit、docs/graph  ← 與 A 無衝突
├── C  ...             等 A、B、決定
├── D  ...             等 A、C
├── E  ...             ⚠ 與 B 共用 C_PAPER_CONTRADICTIONS.md
└── F  ...             等 A 讓出 GPU
```

**FILESYSTEM CONFLICT RISK：E 與 B 都會改 `docs/audit/C_PAPER_CONTRADICTIONS.md`。**
B 未經 Master 驗收前，E 不得 fork。

---

## 等待 Kyzen 裁決的研究決定

| # | 決定 | 擋住 |
|---|---|---|
| D-α | τ = 0.5（論文明文）還是維持 0.07 可學 | C → D |
| D-β | U-16 兩塔共享模式（B 完成後裁決） | C → D |
| D-γ | 3 筆 v1 殘留標註如何處理 | C |
| D-δ | S2 是否翻案（前提已被 Master 部分推翻） | E |
