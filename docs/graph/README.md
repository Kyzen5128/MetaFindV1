# MetaFind 復現 — 設計文件索引

依 `graph-engineering` skill 的 15 步流程，為「在單張 RTX 4090 上復現 MetaFind」
產出的完整 graph specification。**目前狀態：設計完成，尚未動工，等待確認。**

## 閱讀順序

| # | 檔案 | 內容 | 先讀這個如果… |
|---|---|---|---|
| 1 | [`00_FINDINGS.md`](00_FINDINGS.md) | 實際檢查論文、ULIP、egnn、本機環境後的 **9 項硬事實**，以及由此推導的 **3 個架構決策** | 你想知道「為什麼是這樣設計」 |
| 2 | [`01_GRAPH_SPEC.md`](01_GRAPH_SPEC.md) | §15 的 15 項產出：分類、目標邊界、state、節點、邊、相依、路由、迴圈、失敗、驗證、gate、可觀測性、圖、執行順序、風險 | 你要審查設計本身 |
| 3 | [`02_BUILD_STEPS.md`](02_BUILD_STEPS.md) | 可執行版本：專案結構、逐步驟做什麼、**每步的通過條件**、進度追蹤表 | 你要開始動手寫 code |
| 4 | [`graph_spec.yaml`](graph_spec.yaml) | 結構化：45 個 state channel、50 條邊、16 組 join policy、12 個決策點、4 個 cycle | 你要餵給工具或做程式化檢查 |
| 5 | [`node_registry.yaml`](node_registry.yaml) | 結構化：27 個節點 + 4 個 subgraph 的完整 registry（含逐節點 failure policy） | 同上 |
| 6 | [`validation_plan.yaml`](validation_plan.yaml) | 結構化：47 個 L1、15 個 L2、5 個 gate、3 個 Required Audit | 同上 |

## 一頁摘要

**Graph 分類**：`hierarchical DAG + stateful + parallel`，主線零回邊；
3 個 cycle 全封在 subgraph 內（標註修復、語意邊修復、Algorithm 1 迭代組合）。

**三欄 taxonomy**：`control_authority: A1` ／ `execution_mode: probabilistic` ／ `topology_class: workflow`
—— GPT-4o 出現三次但**從不決定路由**，只產生 payload。

**規模**：27 節點、4 subgraph、5 gate、62 個測試、20 個執行層、預估 2–4 週。

### 三個架構決策（全部由環境事實逼出來的）

| | 決策 | 為什麼 |
|---|---|---|
| **D1** | 不重訓 ULIP-2，用官方 checkpoint 當 frozen backbone | 單卡 24GB vs 官方腳本假設的 8 卡 |
| **D2** | **把 frozen backbone 輸出預先算好快取**，訓練只在 1280-d 向量上進行 | 同時解決：單卡可行、ablation 成本降兩個數量級、磁碟壓力解除 |
| **D3** | Shard 串流（刪原始檔改為**可選**） | 改用 `/mnt/data1`（779GB）後強制刪檔已不需要；串流保留是為了續跑粒度 |

### 最需要你確認的五件事

| | 項目 | 為什麼需要你決定 |
|---|---|---|
| 1 | **論文 §2.5 與 Appendix C 自相矛盾**（`h⁰=Concat(x,t)` 破壞等變性，見 F1） | 我的處置是：主線用正確實作，字面實作走 **Required Audit RA-1 並允許它失敗**。這牽涉到你想怎麼呈現這個發現 |
| 2 | ~~磁碟~~ **已解決** | 大型資料改放 `/mnt/data1/kyzen/MetaFind`（779GB），repo 內 `./data` symlink 指向 |
| 3 | **GPT-4o 標註估 US$1.5–2.5K**（F7） | 有免費替代方案（ULIP-2 現成 captions），代價是文字分佈與論文不同。`budget_cap` 要你定 |
| 4 | **5 位專家 × 200 場景 ≈ 67 人時**（G4） | 這是 Table 2 人工欄的必要成本。找不到 5 個人的話，該欄只能判 `INSUFFICIENT_EVIDENCE` |
| 5 | **U-04：Table 1 的 gallery 分母未知** | 唯一一個「猜錯會讓整張 Table 1 失去意義」的未知。目前設計是鎖成 `write_once` 並在報告明列 |

### 已知會偏離論文之處（都會寫進報告）

- **D1**：用釋出的 ULIP-2 權重，非自行預訓練 → Table 1 絕對數值可能偏移（SC-1 的 ±3pp 容差已考慮）
- **D2**：凍結 backbone → Table 3 第 9 列「full fine-tune > fuser only」無法完整驗證（RA-3）
- **U-02**：Table 3 場景數論文未寫，暫用 50（Table 2 用論文明載的 200）

---

## 設計上刻意做的幾個判斷

**gate 只有 5 個，但測試有 62 個。** 被降級的 gate 候選有 6 個，都列在
`validation_plan.yaml: rejected_gate_candidates` 裡並寫明「不符合四判準的哪一條」。

**三個 Required Audit 是「必跑、必留紀錄、但永不阻斷」的檢查。**
其中 RA-1 與 RA-3 **預期會失敗** —— 它們失敗的唯一合法後果是縮小某個 claim 的範圍。
把它們設成 gate 會導致一種很糟的結果：為了讓檢查變綠而放寬判準。

**`n19_aggregate_tables` 用兩個不同政策的 join_group**：
`core`（Table 1、Table 2 的 GPT-4o 欄）用 `all` 必須齊全；
`extended`（人工欄、Table 3）用 `all_settled` 可以部分缺。
一個標註者請假不該擋掉整份報告，但 Table 1 缺格必須擋。

**SC-3 是一個「期待模型變差」的成功判準**：論文 Table 1 的 PC-Only 欄，
MetaFind（63–75）**低於** baseline（98–99）。復現論文包含復現它的退化。

---

*設計依據：`~/.claude/skills/graph-engineering`。
參考實作：[`/home/kyzen/ULIP`](file:///home/kyzen/ULIP)、[`/home/kyzen/egnn`](file:///home/kyzen/egnn)。*
