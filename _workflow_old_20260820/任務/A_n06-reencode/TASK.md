# TASK — A_n06-reencode

> 這份 TASK.md 是本 branch 的 **execution contract**。
> 支線不得自行改寫 Objective / Scope / Explicit Non-Scope / Definition of Done。
> 需要修改請回報 Master。

## Branch Name

`A_n06-reencode`

## Task ID

M1（`主線.md`）／ T4.3（`TASKS.md`）／ node `n06_encode_text_image` + `n05b_resolve_stage1_encoding`

## Objective

用目前的**公分版** `TEXT_TEMPLATE` 對全部 45,952 筆 v3 標註重新編碼 text / image embeddings，
並確保 `stage1_encoding_protocol.json` **誠實記錄本次實際使用的模板**。

## Why Now

- GPU 現在閒置（RTX 5090 32GB，實測 106 MiB used）。
- M1 擋住 M2 smoke、M3、M4、M5、M6。
- 本任務內部**零研究決定**：τ、U-16、3 筆殘留都只擋 n09，不擋 n06。

## Current Context

Master 於 2026-08-20 實測驗證：

- `data/outputs/annotations/` 有 45,955 個 json = **v3 45,952 + v1 殘留 3**。
- 現有 n06 快取只有 **5,276 組**（json+npz），且是**公尺**舊模板。
  實測樣本：`"... roughly 0.25 by 0.15 by 0.05 metres, typically placed floor."`
- `metafind/models/resolve_stage1.py:96-100` 的 `TEXT_TEMPLATE` 已改為**公分**版。
- **但** `data/outputs/stage1_encoding_protocol.json` 記的仍是舊的公尺模板。
- `metafind/data/encode_text_image.py:191` 直接 import `serialize_annotation()`，
  `load_protocol()` **只檢查** `status` / `actual_clip_train_scope` / `image_aggregation`，
  **完全不比對 `text_template`** —— 所以不先修紀錄檔就跑，provenance 會是錯的。
- 3 筆 v1 殘留會讓 `serialize_annotation()` 丟 `KeyError: 'width'`（Master 已實測）：
  - `a397b648d6eb48d7909d1ee11235e78f`（aircraft carrier，mass 50000kg 密度檢查失敗）
  - `6c7db00cc164467ebac356a5ca67368b`（`pom.pom.n.01` synset 格式錯）
  - `8a0192eee6fb4140bb3e9696b3dbae5a`（同上）
  `encode_text_image.py:213-221` 會安全隔離它們。**這是預期行為，不是 bug。**
- 起點：`git HEAD = 4a4ebbe`，工作樹乾淨，`pytest tests/ -q` = 442 passed，
  `tools/check_graph.py` = 2,275 checks all pass。

## Authoritative Inputs

1. `docs/paper/metafind_source/2methdology.tex` + `data-preprocess.png`（13 欄位 schema，Figure 2）
2. `docs/graph/node_registry.yaml` 的 `n05b_resolve_stage1_encoding`（L251）與 `n06_encode_text_image`（L306）
3. `metafind/models/resolve_stage1.py`
4. `metafind/data/encode_text_image.py`
5. `data/outputs/annotations/`、`data/outputs/logs/renders_index.jsonl`

## Dependencies

M0（n05 v3，已完成）。**無其他前置。**

不依賴 B、C、D、E、F。

## Scope

1. **備份比對用檔案**
   ```bash
   cp data/outputs/stage1_hyperparameters.json /tmp/hp_before.json
   cp data/outputs/stage1_encoding_protocol.json /tmp/proto_before.json
   ```
2. **重跑 n05b，讓紀錄檔誠實**
   ```bash
   /home/kyzen/miniconda3/envs/MetaFind/bin/python -m metafind.models.resolve_stage1 \
     --paper-clip-train-scope frozen \
     --actual-clip-train-scope frozen \
     --confidence moderate \
     --decided-by "Kyzen (relayed GPT analysis, 2026-08-16)"
   ```
   三個值＝磁碟現值，**不引入新研究決定**。`--decided-by` 保留原 provenance 字串是**刻意的**。
3. **確認 hyperparameters 值未變**：與 `/tmp/hp_before.json` 逐欄位比對，
   `init_temperature` 必須仍是 `0.07`。不同就**停下回報**。
4. **全量 n06，用 `--force`，不要 `rm -rf`**
   ```bash
   nohup /home/kyzen/miniconda3/envs/MetaFind/bin/python \
     -m metafind.data.encode_text_image --force \
     >> data/outputs/logs/n06_full.log 2>&1 &
   ```
   `is_complete()` 只看 `encoder_version` 與檔案存在，**不看模板** —— 這正是必須 `--force` 的原因。
   `--force` 與 `rm -rf` 結果相同，但不做破壞性刪除、中斷可續。約 4 小時。
5. 完成後執行下方 Required Verification 全部條目。
6. 記錄 3 筆 v1 殘留進入 n06 quarantine 的事實。

## Explicit Non-Scope

- 不決定 τ（保持 `0.07`）— 那是 D-α，Kyzen 的裁決
- 不決定 `tower_sharing` — 那是 D-β，B 支線的範圍
- 不處理 3 筆 v1 殘留：**只記錄，不刪、不改、不補標註** — 那是 D-γ
- 不跑 n09 / Stage 1 / smoke
- 不動 `data/outputs/annotations/`、`annotations_v1_prompt1/`、`annotations_v2_sample/`
- 不動 ESSGNN / stage2 任何檔案
- 不改 `metafind/` 任何原始碼
- 不改 `主線.md` / `支線任務.md` / `TASKS.md`
- **不碰 `docs/` 底下任何檔案** — B 支線正在寫那裡
- 不改其他支線的 `TASK.md` / `HANDOFF.md` / `CODEX_REVIEW.md`
- 不宣告「M1 完成」、不推進到 M2

## Expected Deliverables

1. `data/outputs/embeddings/` 45,952 組 `.json` + `.npz`
2. 更新後的 `data/outputs/stage1_encoding_protocol.json`（公分模板）
3. `data/outputs/logs/n06_full.log`（本次 append）
4. n06 quarantine 紀錄，含那 3 筆
5. `任務/A_n06-reencode/HANDOFF.md`

## Likely Files

以唯讀執行為主。可能寫入：

- `data/outputs/embeddings/`
- `data/outputs/stage1_encoding_protocol.json`
- `data/outputs/stage1_hyperparameters.json`（值不變，僅時間戳）
- `data/outputs/variant_registry.json`
- `data/outputs/logs/`

**不應有任何 `metafind/` 原始碼變更。若出現，立刻停下回報 Master。**

## Required Verification

每一條都要有**實測輸出的數字**，不要只寫「通過」。

- [ ] `ls data/outputs/embeddings/*.npz | wc -l` == `45952`
- [ ] `ls data/outputs/embeddings/*.json | wc -l` == `45952`
- [ ] 隨機抽 5 個 sidecar，`text` 含 `centimetres` 且**不含** `metres,`
- [ ] 每個 `.npz` 有 `text(1280,)` / `image(1280,)` / `views(11,1280)`
- [ ] `text_truncated == true` 的筆數 — **回報實際數字，非零不得就地改程式**
- [ ] n06 quarantine 恰 3 筆，uid 與 Current Context 所列相符
- [ ] `stage1_encoding_protocol.json` 的 `text_template` == `resolve_stage1.TEXT_TEMPLATE`
- [ ] `stage1_hyperparameters.json` 的 `values` 與 `/tmp/hp_before.json` 逐欄位相同
- [ ] `python -m pytest tests/ -q` 仍 `442 passed`
- [ ] `python tools/check_graph.py` 仍 `2275 checks / all pass`
- [ ] `git status --short` 無 `metafind/` 變更
- [ ] 抽樣診斷：同類別資產 text 向量 cosine vs 不同類別（**回報數字，這是診斷不是門檻**）

## Research Risks

- 重跑 n05b 會刷新 `decided_at` 時間戳。**必須在 HANDOFF 中明載**，否則 provenance 看起來像新決定。
- `text_truncated` 若大量出現，代表公分模板變長把 placement 尾巴截掉。
  **這是研究發現，要回報，不是就地修模板** —— 模板是 U-15 的 IMPLEMENTATION CHOICE，改它需 Kyzen 裁決。
- 舊 5,276 筆若沒被 `--force` 全部覆蓋，會留下混合模板的語料。計數驗收就是為了抓這個。

## Implementation Risks

- 4 小時長跑；用 `nohup` 背景執行並可續跑。
- ViT-bigG-14 9.5 GB，須確認 `paths.setup_env()` 的 `HF_HOME` 在 import transformers/open_clip **之前**生效。
- 磁碟：11 個 1280-d float16 / 資產 ≈ 28 KB，全量約 1.3 GB。跑前確認空間。

## Codex Review Requirement

**LIGHT。** 本任務是既有程式的一次執行，非架構或研究判讀。

**僅在下列情形升級為 FULL 並執行 `codex-reviewer`：**

- (a) 需要修改任何 `metafind/` 原始碼
- (b) `text_truncated` 非零
- (c) quarantine ≠ 3

升級時，Codex 的每一項 finding 必須由 Claude 分類為
`CONFIRMED` / `PLAUSIBLE` / `REJECTED` / `UNVERIFIED`，附分類理由，寫入 `CODEX_REVIEW.md`。
**Codex 不是科學權威。**

## Definition of Done

上列 Required Verification 全部逐條有實測輸出，且無任何 Explicit Non-Scope 項目被觸碰。

**Branch 不得自行宣告「M1 完成」或推進到 M2。** 只回報，由 Master 裁決。

## Return-to-Master Requirements

寫入 `任務/A_n06-reencode/HANDOFF.md`，至少包含：

1. Task ID / Status
2. Objective Result
3. Files Changed（`git diff --stat` + `data/` 的實際變動描述）
4. Evidence Used
5. Decisions Made（**應為「無研究決定」**，若非如此必須逐條列出）
6. Verification Performed / Verification Result（逐條實測數字）
7. Codex Review Result（LIGHT 未觸發則寫「未觸發，理由」）
8. Confirmed / Rejected / Unverified Findings
9. **Master-Impacting Findings** —— 明確一行：「是否發現任何會改變 master assumption 的證據」，
   有則列出並標 evidence class（PAPER FACT / UPSTREAM FACT / OBSERVED IMPLEMENTATION /
   OBSERVED DATA / INFERENCE / IMPLEMENTATION CHOICE / DEVIATION / UNKNOWN）
10. Remaining Risks / Blocked Items
11. Recommended Master Update
12. Recommended Next Action

另需附：逐字的完整執行指令（含所有旗標）、git commit SHA（跑前／跑後）、
wallclock、GPU 峰值記憶體、`text_truncated` 筆數（非零時附 3 個範例文字）、
quarantine 完整內容、`stage1_encoding_protocol.json` 的 before/after diff。
