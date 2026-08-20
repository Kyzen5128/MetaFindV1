# MetaFindV1 — 待修項目（依相關性分任務）

**建立於 2026-08-20。**

分任務的原則：**同一個任務裡的項目共用一個根因、一次重跑、或一個研究決定。**
不同根因的東西分開，方便逐任務審查與驗收。

每個任務都寫明：為什麼是一個任務、逐項證據、分類、驗收方式、重跑成本、阻擋關係。

分類詞彙：

| 詞 | 意思 |
|---|---|
| `RESEARCH-SIGNIFICANT BUG` | 會改變科學結果，或讓錯誤無法被發現 |
| `STALE ASSUMPTION` | 建立在已被推翻的判讀上 |
| `INFORMATION LOSS` | 手邊有更好的資料卻沒用 |
| `PAPER-FIDELITY FIX` | 讓實作更貼近論文明文 |
| `DOCUMENTATION-ONLY` | 只影響可讀性，不影響執行 |
| `BLOCKED BY DECISION` | 等人做研究決定才能動 |

---

## 已完成（2026-08-19，僅供對照，不需再審）

| 項目 | 檔案 |
|---|---|
| 資料根目錄統一由 `paths.py` 供應 | `metafind/paths.py` + 6 支 shell |
| `01_storage.sh` 重寫（原本建的目錄結構與程式實際用的不同） | `setup/01_storage.sh` |
| 論文圖檔解壓 + manifest 記錄 `missing_figures` | `tools/build_source_manifest.py`、4 個 `SOURCE_MANIFEST.json` |
| τ = 0.5 列為 PAPER FACT，偏離時警示 | `metafind/models/losses.py`、`docs/audit/C_PAPER_CONTRADICTIONS.md` |
| 協定讀取端實際試建，`resolved` 不再是無檢查的宣稱 | `metafind/train/stage2.py` |
| `status.sh` 依 `prompt_version` 計數 | `tools/status.sh` |

---

# T1 — ESSGNN 架構軸重構

**為什麼是一個任務：** 全部 12 處都是同一個根因的下游 ——
`docs/graph/02_BUILD_STEPS.md:854` 把 U-26 登記成**兩個問題綁在一起**
（「是否共用一條訊息」＋「`f_x` 看到 `h^{l+1}` 還是 `h^l`」）。
程式後來把它拆成兩個旋鈕，卻**選錯了主軸**。改一處而不改其他處會產生自相矛盾的狀態。

**依據：** `essgnn.md`（本 repo），外部 adjudication 已確認
`A = UNSUPPORTED`（架構家族分類無論文證據）、`B = PROVABLY DIFFERENT`（公式確實不同，承重理由是 `h^{l+1}` 那個上標）。

## 逐項

| # | 位置 | 現況 | 分類 |
|---|---|---|---|
| 1.1 | `docs/graph/02_BUILD_STEPS.md:854` | **根因。** U-26 一個編號兩個意義；`graph_spec.yaml:602` 又把 U-26 寫成 coord_feat | `STALE ASSUMPTION` |
| 1.2 | `metafind/models/essgnn.py:191-195` | `__post_init__` 由 `architecture_family` 推導 `coord_feat` | **`RESEARCH-SIGNIFICANT BUG`** |
| 1.3 | `metafind/models/essgnn.py:491-503` | 選 `appendix_shared_msg` 就強制 `coord_feat="current"`，否則 raise | **`RESEARCH-SIGNIFICANT BUG`** |
| 1.4 | `metafind/models/essgnn.py:90-91` | 註解「different parameter counts, different gradient paths」—— 那是**我們實作**的性質，不是論文的 | `STALE ASSUMPTION` |
| 1.5 | `metafind/models/essgnn.py:154-165` | `<- primary` 措辭把 INFERENCE 講得像權威 | `STALE ASSUMPTION` |
| 1.6 | `metafind/models/resolve_stage2.py:99-131` | ARCH_DECISIONS 的 C1 決策理由 | `STALE ASSUMPTION` |
| 1.7 | `docs/audit/E_GRAPH_REVALIDATION.md:173` | **最嚴重的文件錯誤。** 標 `VERIFIED`，理由卻是「§2.5 有 `f_h`/`f_x`，附錄有 `φ_e`/`φ_x`/`φ_h`」—— 那正是被禁止的符號論證 | `STALE ASSUMPTION` |
| 1.8 | `docs/audit/C_PAPER_CONTRADICTIONS.md:25-77` | C1 標題「two different ESSGNNs — STRUCTURAL, blocking」 | `STALE ASSUMPTION` |
| 1.9 | `docs/graph/00_FINDINGS.md:1091` | 「正文與 Appendix C 描述的是兩個不同的架構」 | `STALE ASSUMPTION` |
| 1.10 | `docs/graph/README.md:17` | U-26 結論措辭 | `DOCUMENTATION-ONLY` |
| 1.11 | `tests/test_resolve_stage2.py:179` | 硬斷言 `== "appendix_shared_msg"`，把 INFERENCE 鎖進 CI | `STALE ASSUMPTION` |
| 1.12 | `tests/test_essgnn.py:24,28` | `FAMILY` / `TWO_MLP` 的 primary / competing hypothesis 措辭 | `DOCUMENTATION-ONLY` |

## 為什麼 1.2 / 1.3 是 BUG

```python
# essgnn.py:191-195
if self.coord_feat is None:
    self.coord_feat = ("current"
                       if self.architecture_family == "appendix_shared_msg"
                       else "updated")
```

**有論文依據的軸（`coord_feat`）成為無證據的軸（`architecture_family`）的附屬品。**
結果：**無法在固定其他條件下單獨測 `coord_feat`** —— 而那是唯一該測的維度。

實測三個變數同時不同（違反 `.claude/rules/experiments.md` §2「一個實驗一個問題」）：

| family | 參數 | h→gate 線性層數 | coord_feat |
|---|---|---|---|
| `appendix_shared_msg` | 164,737 | 4 | `current` |
| `sec25_two_mlp` | 213,761 | 2 | `updated` |

**只有第三欄有論文依據。** 深度 4 vs 2 論文完全沒提，是實作引入的。

## 目標狀態

```
主軸（唯一 paper-conflicted 維度）
    coord_feat: "current"  = gate 讀 h^(l)      附錄 Eq.13
                "updated"  = gate 讀 h^(l+1)    §2.5 Eq.3

次要 ablation（標為 IMPLEMENTATION ABLATION，不是 PAPER ARCHITECTURE FAMILY）
    shared-message vs independent heads

必須固定：sharing 結構、MLP 深度、hidden 寬度、參數量、初始化、distance、層數
```

> **用詞注意：`coord_feat` 是 `paper-conflicted`，不是 `paper-backed`。**
> 兩個值都是論文寫的，論文自己打架。

## 驗收

- [ ] `ESSGNNConfig(architecture_family=X, coord_feat=Y)` 四種組合都能建構，無強制
- [ ] 兩個 family 在固定 `coord_feat` 下參數量與深度相同（可比）
- [ ] `tests/test_essgnn.py` 的等變測試對四種組合都跑
- [ ] 移除 `tests/test_resolve_stage2.py:179` 的硬斷言
- [ ] 全 repo 搜 `two different ESSGNNs` 零命中
- [ ] `tools/check_graph.py` 全過

**重跑成本：** 無（尚未訓練過 ESSGNN）
**阻擋：** 無。可獨立進行。
**狀態：** `BLOCKED BY DECISION` —— 需你確認翻案。

---

# T2 — n08 節點文字資訊塌縮

**為什麼是一個任務：** 單一根因（`object_text()` 只用 category），單一重跑（n08，實測 22 分鐘）。
與 T3 分開，因為 T3 要動 n07 且會改圖拓樸。

## 證據

```
assetId 數                  1,467
不重複 node text 數            93     ← object_text() 只吃 category
C(93,2)+93 理論上限          4,371
實際不重複配對                4,242    ← 佔理論上限 97.0%
若以 assetId 去重（前1500房） 114,417
```

`metafind/data/scene_graphs.py:96-104`：

```python
def object_text(category: str) -> str:
    return f"a {humanise(category)}"
```

`Chair_1` … `Chair_50` 全部變成 `"a chair"`。程式自己也承認（`scene_graphs.py:40-43`：「every instance of a category shares it. **That is a ceiling.**」，登記為 U-12）。

## 手邊有、卻沒用的真值

`outputs/procthor_modalities/*.json` 與 AI2-THOR 執行時 metadata：

| 欄位 | 內容 | 覆蓋 |
|---|---|---|
| `bbox_reported` | **真實公尺尺寸**，與我們自量的深度殼差 5.6 mm | 1,467 / 1,467 |
| `salientMaterials` | 受控詞彙 15 個（Plastic/Metal/Wood/Glass/Fabric/Food/…） | AI2-THOR 查詢可得 |
| `pickupable` / `receptacle` | 真實 affordance | 同上 |
| `parentReceptacles` | 真實擺放對象 | 同上 |

## 逐項

| # | 位置 | 動作 | 分類 |
|---|---|---|---|
| 2.1 | `metafind/data/scene_graphs.py:96-104` | `object_text()` 納入 AI2-THOR 真值（尺寸／材質／affordance） | `INFORMATION LOSS` |
| 2.2 | `metafind/data/scene_graphs.py` | 新增擷取步驟把 AI2-THOR metadata 寫進 `procthor_object_text.json` | `INFORMATION LOSS` |
| 2.3 | 舊稽核紀錄 | 「列出所有在同一個 house 共現的物件 pair」→ 實際是 `support ∪ kNN(k=8)`（`scene_graphs.py:210`） | `DOCUMENTATION-ONLY` |
| 2.4 | 舊稽核紀錄 | 「25% 的關係是錯的」→ 只能說「25% 含 might/could/may/often 這類詞彙標記，不能推斷正確性」 | `DOCUMENTATION-ONLY` |

## 不要做的

**不要加 no-relation / 過濾。** 論文 §2.3 只說 "obtained by prompting an LLM on object pairs"，
**完全沒提過濾**。加過濾是發明，而且會改 `edge_index` 拓樸（`stage2.py:176` 直接迭代 `sem_edge_ids`）。

## 驗收

- [ ] 不重複 node text 數 > 93（預期數百至千級）
- [ ] 不重複配對數顯著大於 4,242
- [ ] `sem_edge_embeddings.npz` 維度與 `sem_edge_cache.json` 宣告一致
- [ ] 零全零向量、零降級
- [ ] 抽樣人工檢視關係句是否比 category-only 版具體

**重跑成本：** n07 文字重建（分鐘級）＋ n08（實測 22 分鐘，需 GPU）
**阻擋：** 需等 n05 讓出 GPU
**狀態：** `BLOCKED BY DECISION` —— node text 要納入哪些欄位是研究決定

---

# T3 — n07 丟棄 ProcTHOR 的支撐方向

**為什麼與 T2 分開：** 這會改**圖的拓樸語意**，牽動 ESSGNN 的訊息傳遞，
且需要 n07 + n08 都重跑。T2 只換文字，不動邊。

## 證據

ProcTHOR 的 `children` 樹提供 parent→child 方向，`scene_graphs.py:155` 有讀到：

```python
support.append((parent, idx))          # 方向存在
```

但下一步就丟掉：

```python
# scene_graphs.py:203
support = sorted({(min(a, b), max(a, b)) for a, b in support_directed})
```

之後 `semantic_edges_run.py:139` 又按**字母序**排一次：

```python
a, b = sorted((ti["text"], tj["text"]))
```

所以句子的主賓順序由字母決定，**與真實支撐關係無關**。
`stage2.py:185-187` 雙向展開並複製同一個 `e_ij`。

## 現況判定

**不是 BUG。** U-19 已宣告「無向」為我們的慣例，且從 n07 → n08 → stage2 全程一致，
無非決定性問題。**但是可回收的 `INFORMATION LOSS`。**

## 逐項

| # | 位置 | 動作 | 分類 |
|---|---|---|---|
| 3.1 | `metafind/data/scene_graphs.py:203` | 保留 `support_directed`，另存為有向欄位（不覆蓋現有無向欄位） | `INFORMATION LOSS` |
| 3.2 | `metafind/data/semantic_edges_run.py:139` | 主賓順序改由支撐方向決定，而非字母序 | `INFORMATION LOSS` |
| 3.3 | `metafind/train/stage2.py:185-187` | 反向邊是否該用不同 `e_ij` | `BLOCKED BY DECISION` |

## 論文依據

**論文對邊的方向完全沉默**（U-19）。§2.3 只說兩種邊，§2.5 只有一個 `e_ij`。
**所以 3.1–3.3 全部是 IMPLEMENTATION CHOICE 的另一個值，不是 PAPER-FIDELITY FIX。**

## 驗收

- [ ] 有向資訊存在 artifact 中且可還原
- [ ] 現有無向路徑不受影響（可回退對照）
- [ ] 若採用有向 `e_ij`：`edge_attr` 列數 = `edge_index` 行數，且正反向不同

**重跑成本：** n07（分鐘級）＋ n08（22 分鐘）
**阻擋：** 應在 T2 之後或與 T2 合併重跑，避免跑兩次
**狀態：** `BLOCKED BY DECISION`

---

# T4 — n05 收尾與 n06 重編碼

**為什麼是一個任務：** n05 的產出直接決定 n06 要編碼什麼，兩者必須連著做。

## 逐項

| # | 動作 | 說明 |
|---|---|---|
| 4.1 | n05 全量跑完後檢視隔離紀錄 | 確認新加的密度檢查（`annotate.py`，mass/w·l·h）沒有誤殺。目前 33,600 筆隔離 3，率 0.009% |
| 4.2 | 用 3,289 個 AI2-THOR 真值資產驗收 `onFloor` / `onObject` | baseline：v1 = 67.2%（多數類 58.0%）；v3 樣本 = 72.5% / 71.0% |
| 4.3 | **丟棄現有 5,276 筆 n06 快取，全量重編碼** | 文字模板已改（公尺→公分、placement 改四布林），舊向量與新標註不一致 |
| 4.4 | 決定 `TEXT_TEMPLATE` 是否納入 `synset` / `volume` / `mass` | 目前三者留在標註檔但不進 CLIP。論文未規定序列化。 |

## 已知的天花板（要寫進限制章節，不是待修項）

`onFloor` / `onObject` 需要知道物件多大，而 **n04 的尺度正規化讓圖上沒有絕對大小**。
實測錯誤是系統性的：大型家具被誤標 `onObject`、小型可攜物被漏標 `onFloor`。

**n04 的正規化是必要且正確的** —— Objaverse 的單位混亂（實測 47% 超過 10，常見值 1000/100/200），
不正規化根本渲染不出來。**這是資料集本身的限制，不是可修的 bug。**

## 驗收

- [ ] n05 隔離率 < 2%（G3 門檻）
- [ ] `onFloor` / `onObject` 正確率 ≥ v1 baseline
- [ ] n06 embeddings 數 = n05 標註數
- [ ] `serialize_annotation` golden string 測試通過

**重跑成本：** n05 剩約 5 小時；n06 全量約 4 小時
**阻擋：** T4 完成前 Stage 1 不能跑
**狀態：** 進行中（n05 33,600 / 45,556）

---

# T5 — Stage 1 啟動前置

**為什麼是一個任務：** 這幾項全部是 Stage 1 `main()` 開頭就會撞的門檻，缺一不可。

## 逐項

| # | 項目 | 現況 |
|---|---|---|
| 5.1 | `n09_build_splits` **從來沒跑過** | `splits.json`、`eval_protocols.json`、`stage1_protocol.json` 三個檔案都不存在，`stage1.py:356` 直接 return 2 |
| 5.2 | `stage1_hyperparameters.json` 的 τ | 目前 `init_temperature=0.07, learnable=True`。論文是**固定 0.5**（見已完成項）。要決定主線用哪個 |
| 5.3 | U-16（兩塔是否共享權重） | **架構圖有新證據**：`MetaFind.drawio.png` 標著 `ULIP-2 (Shared)`，且只畫一個 Fusion Layer |
| 5.4 | `Stage1RuntimeConfig` 宣稱是唯一建構路徑，但 `stage1.py:309-338` 不用它 | 34 條 `test_dual_tower.py` 測的是訓練器繞過的那個 class |

## 5.3 的新證據與新矛盾

架構圖畫的是**一個** Fusion Layer，在標著 `Shared` 的框裡。
但 §2.6 說「Only the query-side fusion layer and the ESSGNN module are updated; the gallery encoder is frozen」。

**若 fuser 真的只有一個模組，「凍結 gallery」與「訓練 query fuser」不可能同時成立。**
`dual_tower.py:315-321` 正是因此在 `fully_shared` 下拒絕 `freeze_gallery()`。

**這是新的 C 系列矛盾候選，尚未登記。**

## 驗收

- [ ] `n09_build_splits` 產出三個檔案，`tools/check_graph.py` 的 G3 判準通過
- [ ] τ 決定寫進 `stage1_hyperparameters.json` 並記錄分類
- [ ] U-16 決定寫進 `stage1_protocol.json`，附架構圖證據
- [ ] `stage1.py` 走 `Stage1RuntimeConfig`，或明確記錄為何不走
- [ ] Stage 1 smoke（200 資產、1 epoch）產出完整三段 checkpoint

**重跑成本：** n09 秒級；smoke 約 10 分鐘
**阻擋：** 需 T4 的 n06 完成
**狀態：** `BLOCKED BY DECISION`（τ、U-16）

---

# T6 — Table 2 評估管線

**為什麼是一個任務：** 五個節點完全沒有程式，且共用同一組輸入（200 個場景）與同一套評分。

## 現況

```
n15a_resolve_eval_scene_protocol   只有規格
n15b_resolve_composition_protocol  只有規格
n15c_prepare_eval_scenes           只有規格
n16_compose_scenes                 只有規格
n17_judge_scenes                   只有規格
```

## 新取得的依據（本輪讀 I-Design 原始檔取得）

| 項目 | 內容 |
|---|---|
| **60 條 prompt** | `idesign_source/tabs/tab_promptlist_{minimal,others}.tex`，四種型態，**每條附房間三維尺寸** |
| **11-20 條「指定佈局」型** | 直接對應 Algorithm 1 —— prompt 內即含初始佈局 |
| **評分維度出處** | MetaFind Table 2 的前兩維與 I-Design **逐字相同**，後兩維換掉。MetaFind 有引用 I-Design（`ccelen2024design`），provenance 清楚 |
| **⚠️ 尺度不同** | I-Design **0–10**，MetaFind **1–5**。兩篇數字不可直接比較 |
| **評分方式** | I-Design：「兩張不同視角**橫向拼接** + 原始 user input，逐場景單獨評分」。MetaFind 只說「scene layouts and rendered views」 |
| **免 LLM 客觀指標** | `NObj`（平均物件數）、`OOB`（任一 bbox 出界即該場景 invalid）、`BBL`（bbox 交集體積平均）|

## 逐項

| # | 項目 | 分類 |
|---|---|---|
| 6.1 | U-27：200 個場景的構造。**現在有 60 條可引用的 prompt**，差額 140 仍 UNKNOWN | `BLOCKED BY DECISION` |
| 6.2 | n17 評分 prompt 用 MetaFind 的四維，**不是** I-Design 的四維 | `PAPER-FIDELITY FIX` |
| 6.3 | n17 要餵幾張圖／是否拼接／是否附 prompt —— MetaFind 未說 | `BLOCKED BY DECISION` |
| 6.4 | 加入 `NObj` / `OOB` / `BBL`。**不受 D-2（Qwen 換 GPT-4o）與 D-4（無人工評分）污染**，是 Table 2 唯一乾淨的量化支撐 | `QUALITY IMPROVEMENT` |
| 6.5 | 改寫 F18 的失敗歸因 | 見下 |

## 6.5 —— F18 歸因要改

I-Design 的佈局**不是 LLM 出座標**，是：

```
相對場景圖 → 拓樸排序 + cluster 淨空距離 → backtracking 取樣
           → 放不下就回溯到 depth d-1 重新取樣
```

而 I-Design 自報 **OOB = 0.0**（完全零出界）。

現有 F18 記錄「Qwen 跑 5 次 0 個場景完成」。
**對照之下，更可能是 backtracking 求解不出解，而不是 Qwen 失效。** 歸因方向要修。

（`temperature=0.7` / `top_p=1.0` 已正確繼承並記錄於 `tools/idesign_generate.py:15-17`，**非未登記偏離**。）

## 驗收

- [ ] 200 個場景的來源可追溯到 60 條原文，擴充規則明文記錄
- [ ] n17 的 prompt 引用 MetaFind Table 2 的四維與 1–5 尺度
- [ ] `NObj` / `OOB` / `BBL` 可算且與 I-Design 的定義一致
- [ ] 報告明載 I-Design 0–10 與 MetaFind 1–5 不可直接比較

**重跑成本：** 全新實作
**阻擋：** 需 Stage 2 完成
**狀態：** `BLOCKED BY DECISION`

---

# T7 — 用新取得的 39 張圖重查所有 UNKNOWN

**為什麼是獨立任務：** 這不是修某個檔案，是**重新檢驗一個前提**。

## 背景

`docs/paper/*_source/` 原本一張圖都沒有。本輪解壓後：

```
metafind   6 張    ulip2   15 張
egnn       8 張    idesign 10 張        共 39 張
```

**所有既有稽核都是在「只讀文字」的狀態下做的。**
`data-preprocess.png` 印著標註 schema，n05 因此照錯的 schema 做了整整一輪；
`MetaFind.drawio.png` 已確認影響 U-16。

**剩下 37 張還沒逐張看過。**

## 逐項

| # | 動作 |
|---|---|
| 7.1 | 逐張讀 39 張圖，列出每張含有的 PAPER FACT |
| 7.2 | 對照 `docs/graph/` 的 42 項 U 登記，標出：**被圖解決 / 被圖推翻 / 圖沒提** |
| 7.3 | 對照 `docs/audit/` 的 C1–C8、S1–S6 |
| 7.4 | 已知待補登記：架構圖的單一 Fusion Layer vs §2.6 可分別凍結（見 T5.3）|
| 7.5 | 已知待補登記：Holodeck 有 `frontView`，MetaFind Figure 2 沒有 |
| 7.6 | 架構圖的標籤錯誤：`Text Encoder → I1..IK`、`Image Encoder → T1..TK`（字母對調） |

## 驗收

- [ ] 39 張圖每張有一行紀錄：含什麼 PAPER FACT，或「無新規格」
- [ ] U 登記表每一項標明是否被圖影響
- [ ] 新發現的矛盾登記成 C 系列

**重跑成本：** 無（純閱讀）
**阻擋：** 無
**狀態：** 可立即進行。**建議優先於 T1／T5**，因為圖可能再推翻既有判讀。

---

# 任務相依關係

```
T7（讀圖重查）
  │  可能推翻 T1 / T5 的前提
  ├──────────────► T1（ESSGNN 軸重構）        獨立，無重跑
  └──────────────► T5.3（U-16）

T4（n05 收尾 + n06 重編碼）──► T5（Stage 1 前置）──► Stage 1
                                                        │
T2（n08 節點文字）─┐                                     ▼
T3（n07 方向）    ─┴─► 合併重跑 n07+n08 ──────────► Stage 2 ──► T6（Table 2）
```

## 建議順序

1. **T7** —— 純閱讀、零成本，且可能改變 T1／T5 的前提
2. **T4** —— 已在跑，n05 完成後接 n06
3. **T1** —— 不需重跑，可與 T4 平行
4. **T5** —— 需 T4 的 n06
5. **T2 + T3** —— 合併一次重跑，需 GPU（等 n05 讓出）
6. **T6** —— 最後

---

# 不在任何任務裡的（刻意不修）

| 項目 | 為什麼不修 |
|---|---|
| n04 的單位球正規化 | **必要且正確。** Objaverse 單位混亂（47% 超過 10），不正規化渲染不出來。絕對尺度從來沒可用地存在過 |
| n08 加 no-relation 過濾 | 論文完全沒提過濾，加了是發明，且會改 `edge_index` 拓樸 |
| `f_x → R³`（§2.5 字面） | 論文寫錯。實作用純量是對的（實測 R³ 等變誤差 0.43 vs 純量 2.2e-16） |
| D-2 / D-5 / D-7 | 已登記的偏離，非缺陷 |
