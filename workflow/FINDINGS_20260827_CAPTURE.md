# 2026-08-27 跨 session 發現搶救

**寫入者**：MASTER `metafindv1-26 [14a7d3]`，2026-08-27
**為什麼存在**：所有 session 於本日整批重開一次，角色名稱全部改變。
下列發現在重開前只存在於各角色的對話裡，**一個字都沒進過檔案**，
再重開一次就會永久消失。依 `paused-role-findings-die`：停工的角色寫不了檔，
發現要交給寫得了檔的人。本檔即為那次交付。

**每一條都標明出處角色與我這一端的查證狀態。** 未經我查證的一律標 UNVERIFIED，
不因為是同事說的就升級。

---

## 0. 角色名冊（2026-08-27 重開後，全部由各角色從自己的環境讀出回報）

| 角色 | 名稱與 ref | 狀態 |
|---|---|---|
| MASTER | `metafindv1-26 [14a7d3]` | active |
| ULIP2 ENGINEER | `metafindv1-be [9dbe6f]` | STOPPED，無 ✅ |
| ULIP2 REVIEWER | `metafindv1-c4 [e566a3]` | 未被停過，唯讀 |
| ESSGNN ENGINEER | `metafindv1-77 [487717]` | STOPPED，無 ✅ |
| ESSGNN REVIEWER | `metafindv1-9a [fae13b]` | STOPPED，無 ✅ |
| INTEGRATOR | `metafindv1-26 [c7ade6]` | ON HOLD（DL-009）＋ Kyzen 直接停 |

**⚠ `metafindv1-26` 有兩個**：`[14a7d3]` 是 MASTER，`[c7ade6]` 是 INTEGRATOR。
**對外一律帶 ref**，否則會送錯。ROSTER.md:54-66 的舊「兩個 claimant」問題已消解 ——
舊的 `a9/11209` 與 `01/32839` 兩個位址都不在現存 peer 清單中（OBSERVED），
但 ESSGNN REVIEWER 明確表示**它無法從自己這端看到 session 的祖先**，
所以「哪一個是它的前身」的正式答案是「不知道」。不得用推理補。

INTEGRATOR 補充一條先前未記錄的 USER 規則：
**Kyzen 可以在任一角色的視窗打 `✅`，該角色即可帶著轉達，他不需每個視窗重打。**
轉達時必須說明 `✅` 打在誰的視窗、涵蓋哪一次執行。
「一個 `✅` 只涵蓋當次報告、不順延」不變。

**界線（INTEGRATOR 2026-08-27 補正，Kyzen 原話）**：
「我在你這視窗打 ✅ 代表 我同意 你可以給他 不要我來回傳」——
所以有效的是**直接收到該 `✅` 的角色**轉達。
**第三手（A 轉給 B、B 再轉給 C）不在這句話的授權範圍內。** 不得拉成鏈。

---

## 1. 🔴 F-A — `phi_e` 少了 EGNN 明定的尾端 Swish

**出處**：ESSGNN ENGINEER `metafindv1-77`
**我的查證狀態**：**CONFIRMED**，我逐字讀了 EGNN 附錄與我們的程式碼。

EGNN 論文 `docs/paper/egnn_source/sections/appendix.tex:75-79` 逐字：

```
phi_e  Input → {Linear → Swish → Linear → Swish} → Output    ← 尾端有 Swish
phi_x  m_ij  → {Linear → Swish → Linear}         → Output    ← 尾端沒有
phi_h  [h_i, m_i] → {Linear → Swish → Linear → Addition(h_i)} → h^{l+1}
```

同節 `:82` 逐字：**"These functions are used in our EGNN across all experiments."**
→ 這是**跨全部實驗的架構定義，不是 QM9 專用**。屬 **Type A**，不是 Type C。

我們的實作 `metafind/models/essgnn.py:422-426`（`ESSGCLShared`，即現行 primary 家族）：

```python
self.phi_e = _mlp(2 * h + 1 + edge_dim, h, h)   # Linear → SiLU → Linear
self.phi_x = _mlp(h, h, 1)                      # Linear → SiLU → Linear
self.phi_h = _mlp(h, h, h)                      # Linear → SiLU → Linear
```

`_mlp`（`:284`）只有一種形狀：`Linear → SiLU → Linear`。SiLU 與 Swish 是同一個函數。

**逐一對照：**

| 函數 | EGNN 規定 | 我們 | 判定 |
|---|---|---|---|
| `phi_x` | Linear→Swish→Linear | 相同 | ✅ 相符 |
| `phi_h` | Linear→Swish→Linear ＋殘差 | Linear→SiLU→Linear，殘差在外 | ✅ 正確 —— MetaFind Appendix 本來就把殘差放外面逐邊求和 |
| `phi_e` | Linear→Swish→Linear→**Swish** | Linear→SiLU→Linear | ❌ **少尾端 Swish** |

**為什麼 `_mlp` 的 docstring 沒擋住這件事**：它把 U-35 留為 UNKNOWN，
理由是「`f_h` 跟三個都不像」。但那句是針對 **`sec25_two_mlp`** 家族。
我們跑的是 **`appendix_shared_msg`**，它的三個函數就是 EGNN 的 φ_e/φ_x/φ_h，**一對一**。
docstring 的免責在現行家族下不成立。

**尚未確立的**：只驗到「形狀不同」，**未驗「結果會不同」**。
ESSGNN ENGINEER 自己標明此條未經 Reviewer 攻擊。
分類：**EGNN UPSTREAM ARCHITECTURE FACT（Type A）vs OBSERVED IMPLEMENTATION 的落差**。
是否修、怎麼修，需經 Reviewer 與 Kyzen。

---

## 2. F-B — `hidden_dim 128` ＋ `n_layers 4` 在 EGNN 任一實驗中都不存在

**出處**：ESSGNN ENGINEER `metafindv1-77`
**我的查證狀態**：**CONFIRMED**（層數與寬度我先前已各自查過）

```
EGNN N-body        n_layers 4    nf 64     main_nbody.py:29,35
EGNN autoencoder   n_layers 4    nf 64     experiments.tex:153
EGNN QM9           n_layers 7    nf 128    main_qm9.py:30,34 · appendix.tex:135
我們               n_layers 4    nf 128    ← 兩個實驗各取一半
```

`n_layers` 已於 `NOTEBOOK:299` 由 Kyzen 核可改為 7。**`hidden_dim` 仍 OPEN。**

**不得記述為「沿用 EGNN」** —— 沒有任何一個 EGNN 實驗是 4/128。

**補充（ESSGNN ENGINEER，2026-08-27，`.claude/rules/upstream-lookup.md` 的 step 3 已補）**：
我原本只引 argparse 預設，那正是該規則明文記載踩過的坑
（「2026-08-25 — Declared "epochs has no source" ... **Step 3 was skipped.**」）。
它把 launch scripts 與 README 啟動指令查完了：

```
argparse 預設   main_nbody.py:29 --nf 64 · main_ae.py:36 --nf 64 · main_qm9.py:30 --nf 128
repo 內唯一 .sh  n_body_system/dataset/script.sh —— 只產資料，不訓練，不傳 --nf
README 啟動指令  :75 --model tfn --nf 32 · :81 --model se3_transformer --nf 64
                :36 eg.EGNN(hidden_nf=32) 是函式庫用法示例，不是實驗
                **沒有任何一行對 --model egnn 傳 --nf**
→ 三個任務都跑 argparse 預設，無 launch script 覆蓋。64 / 64 / 128 成立。
```

**更銳利的形式**：128 只出現在 QM9，而 QM9 是 7 層。
`main_qm9.py:30` 的 128 與 `:34` 的 7 **是同一組設定的兩半**。
取 128 卻不取 7，是把一組設定拆開。
`n_layers` 核可改 7 之後，我們就是 7/128 = QM9 那一組，
**使 `hidden_dim = 128` 的 provenance 由「不成立」變成「條件成立」。**

⚠ **這不是決定填 128。** `DIM_REVIEW` §7 Step 5 明寫「前四步定了才有合法上游」，
而 Step 1–4（`use_io_projections` / Pooling / `layer_sharing` / edge 維度契約）**一步都還沒定**。
`hidden_dim` 仍 OPEN，`resolve_stage2.py` 仍寫 `None`。

---

## 3. INTEGRATOR 的五條接縫發現

**出處**：INTEGRATOR `metafindv1-26 [c7ade6]`
**我的查證狀態**：**UNVERIFIED**（本輪未逐條開檔驗證，僅原樣保存）
它自述除第 5 條的受害者部分外，皆為它親自讀碼所得的 OBSERVED IMPLEMENTATION。

```
接縫 1  指紋寫了沒接線
        sem_edge_cache.json 有 llm_model（semantic_edges_run.py:391）
        procthor_node_embeddings.json 有 sha256
        但 stage2.py:303-327 只比對向量寬度，那些欄位一個都沒讀
        → 換掉 n08 的模型，訓練照跑不報錯，Table 2 整組數字改變

接縫 2  n08 的 LLM 沒有偏離編號
        semantic_edges_run.py:77，D-2 / D-8 拆分後兩邊都不屬於
        登記簿唯一還開著的洞

接縫 3  tower_sharing="fully_separate" 是空殼
        dual_tower.py:60 定義為兩個骨幹，stage1.py:368 只建一個，且不驗證數量
        → 設下去照跑，跑出來其實是 shared_backbone_separate_fusion

接縫 4  Stage1RuntimeConfig.from_protocols 只有 tests/ 呼叫
        訓練器從未使用（stage1.py:309 直讀原始協定字典）
        → 所有協定驗證在真實訓練路徑上都不執行

接縫 5  FU-A：check_graph.py:373-384 只比對 deviation 的 id
        從不讀 what: / why: / impact: 文字
        → 已知受害者 graph_spec.yaml:1800 的 U-13
          （前任 MASTER 的觀察，對 INTEGRATOR 自己也是 UNVERIFIED）
```

> **⚠ 2026-08-30 MASTER 附註（原文一字未改，這是加註不是改寫）**
>
> 接縫 1 有兩點已經可以觀察到，兩點都讓原措辭站不住：
>
> ```
> OBSERVED  scene_splits.py:102 確實讀了 cache["llm_model"] 與 cache["text_encoder_version"]
>           所以「那些欄位一個都沒讀」太寬。stage2.py:303-327 沒讀，但別的消費者讀了。
>
> OBSERVED  find . -name "*protocol*.json"  在本樹回傳空
>           不論舊措辭「指紋寫了沒接線」或新措辭「協定裡根本沒有那個 key」，
>           講的都是一個這個 repo 目前不存在的檔案。
> ```
>
> **⚠⚠ 2026-08-30 同日再更正：上面那條 `find` 是假的。**
> `data -> /home/kyzen/metafind_data` 是 symlink，`find` 預設不跟隨起點的 symlink，
> 而且**不報錯，安靜回傳空集合**。`find -L` 找得到 7 個協定檔。
> 我用一次因機械原因為空的搜尋做了裁定。**這一條由 ULIP2 Block Reviewer 抓出。**
>
> 重新裁定（證據見 `DL-035`）：
> ```
> 指紋那一半      已關閉   stage2.py:94-99 驗 sha256、:269 讀 llm_model 與
>                          text_encoder_version、:397-398 往下傳。行號已漂移，
>                          原句的 303-327 現在是查詢編碼，不是驗證。
> sem_edge_cache  不適用   find -L 找不到該產物，沒有東西可讀
> 協定缺 key      仍開著   essgnn_arch_protocol.json 沒有 node_feat_dim /
>                          edge_feat_dim。這是 U-20／④，與接縫 1 講的不是同一件事。
> ```
>
> **舊 UNRESOLVED 附註原文保留於下，不刪，因為它本身是證據。**
>
> **UNRESOLVED。沒有就地改寫，因為兩種措辭都還沒被確立，
> 而一個自己都沒查證的更正，正是它宣稱要修的那個錯。**
> 完整脈絡見 `workflow/DECISION_LEDGER.md` 的 `DL-035`。

**接縫 3 與 4 最值得優先驗**：它們讓「設定寫了但不生效」，
正是 Rule 5「不得用檔名／變數名／schema 推論執行期行為」要防的那一類。

---

## 4. ULIP2 ENGINEER 的四個工作單漏洞

**出處**：ULIP2 ENGINEER `metafindv1-be`（回報送給已消失的 `metafindv1-0d`，本檔為搶救）
**我的查證狀態**：漏 1 已由 ULIP2 REVIEWER 獨立複驗（見第 5 節），其餘 UNVERIFIED

```
漏 1  lr 5e-4 / epochs 250 的真正落腳處是 resolve_stage1.py:239,243（現況 1e-3 / 50）
      不是我工作單指的 stage1.py:377。
      stage1.py:377 讀的是 values["learning_rate"]，只改那裡數值不會變。

漏 2  新增的 betas / eps / warmup_epochs / lr_start / lr_end 五欄
      若不進 stage1_config.py:130 的 REQUIRED_HYPERPARAMETERS，
      等於協定有寫、守門沒有。

漏 3  改 DEFAULT_HYPERPARAMETERS 會改變 canonical_hyperparameter_hash
      （splits.py:186 會重算比對，stage1_protocol.hyperparameter_config_hash 也帶著它）。
      正確順序：改 resolve_stage1 → 重跑 n05b → 再跑 n09。
      splits.json / eval_protocols.json / stage1_protocol.json 三個檔目前都不存在，
      所以這次重生成零成本。

漏 4  n09 的 admitted_uids() 交集三個 index。
      annotations_index.jsonl 現在只有 5 行（8/23 smoke 殘留），
      因為 annotate_run.py:1384 的 rebuild_index() 在 main() 結束才跑。
      不是 bug，但代表 n09 必須等 n05 跑完（08-28）才能執行，否則語料只有 5 件。
```

**另外兩點**：

- **`metafind/eval/` 不存在**，repo 裡沒有任何算 R@k 的程式。
  `BLOCK.md:18-19` 的鏈 n10→n10b→n11→G4→n12 中間沒有評估節點。
  評估程式**不是改，是從零建新節點**。
- **實作次序應為 dev-val → optimizer/scheduler → 評估**，不是三件並列。
  依 Rule 10，lr sweep 的評選必須跑在 dev-val 上，dev-val 是前置。

**它推論出、需要裁決的**：`C_dev_selection` 協定的 gallery 是什麼？
它推的是 `gallery = dev_val`（不能是整個 train，否則 dev_val 的排名會被
dev_train 當幹擾項影響，候選池規模跟正式期差一個量級，分數不可比）。
**這是它的推論，不是協定文字。待裁。**

---

## 5. ULIP2 REVIEWER 抓到我兩個錯

**出處**：ULIP2 REVIEWER `metafindv1-c4`
**我的查證狀態**：**CONFIRMED**

```
錯 A  我寫「lr 的落腳處是 resolve_stage1.py:239」—— 行號對，目錄錯。
      metafind/train/resolve_stage1.py     不存在
      metafind/models/resolve_stage1.py    存在，:239 確實是 "learning_rate": 1e-3
      而且這是我先前錯誤 #3 的鏡像 —— 那次我只翻 metafind/models/ 而漏了
      metafind/train/stage1.py。兩個檔各自在我放錯的相反目錄。

錯 B  我的同步讀起來像「08-27 那份清單已經生效」，但程式碼裡沒有。
      metafind/models/resolve_stage1.py DEFAULT_HYPERPARAMETERS 現況：
        learning_rate  1e-3     Kyzen 核可的是 5e-4 起跑
        epochs         50       Kyzen 核可的是 250 上限
        weight_decay   0.1      相符
        batch_size     64       相符 ULIP main.py:51
      不是缺陷，是還沒有人做。但要明說，免得有人讀了以為已落地。
```

**它獨立複驗了我引的上游行號，全部相符**：
`ULIP/main.py:129-135` 參數分組 · `:53-56` lr_start 1e-6 / lr_end 1e-5 ·
`:47` epochs 250 · `:61` eps 1e-8 · `:57` wd 0.1 · `:58` betas (0.9,0.98)。

### 它提出的一個規則內部張力（提出，未解決）

`--epochs default=250` 是 **argparse 預設**。新規則 **Type D** 說
argparse／函式庫預設「永遠不能單獨解決 MetaFind 沉默」。
而 Kyzen 對 250 的記錄理由是「因為 ULIP 有說喔」—— **那正是 Type D 來源。**

**決定本身是穩的** —— Rule 16 承認它，因為 Kyzen 明確核可，那是三條路裡最強的一條。
**問題在記錄的理由**：它引用了新規則剛降級為「永不足夠」的那一類。
若 ledger 寫成「上游說 250」，後來的讀者套 Type D 會把它重開。

→ **應記為 `USER-APPROVED`，上游預設列為佐證而非依據。**
`lr_start 1e-6 / lr_end 1e-5` 同型（我已自行抓到並修正，這是同一形狀往上一列）。

**還有一句要明白寫下**：ULIP 自己的 `--lr` 預設是 **3e-3**，
而 Kyzen 的 sweep「3e-3 不入第一輪」是刻意排除它。那是他的決定，沒問題 ——
但依 Rule 17，**日後任何人都不得把這個 sweep 描述為「照 ULIP 的設定」**。
它照的是 ULIP 的**機制**，並依決定**偏離了它的數值**。

### ULIP2 REVIEWER 自身狀態

最後一次 ULIP2 裁決是 `SharedViewPrefix` 速度批次的 **PASS on the code**
（config-cache、`mm_token_type_ids` guard、`report_low_clip.py`），
外加「不升版」的裁定與理由。**那只清了程式碼，從未清任何一次執行。**
現正在跑的 n05 不是它授權的，它手上沒有 `✅`。

未修的 ULIP2 open findings：
`N-2` `N-5` `C3` `C6` `R-32` `N-3` `N-4` `C5` `C2` `mon`，
外加 `status.sh:50/51` 與 `chain_to_stage1.sh:50,69`。

### 🔴 `chain_to_stage1.sh` —— REVIEWER 的原句已由 ENGINEER 更正，我複驗過

**REVIEWER 原句**：「會在 n05 產出被計數的那一刻觸發，而那已迫在眉睫。」
**ULIP2 ENGINEER 更正 · 我 CONFIRMED**：**它不會自己觸發。**

```
crontab -l | grep chain|stage1     無
pgrep -af chain_to_stage1          沒有在跑
tools/chain_to_stage1.sh:4 的用法註解是 nohup bash tools/... &
→ 手動啟動的腳本，沒有 watcher、沒有 cron。n05 跑完那一刻它不會自己動。
```

**「會自動觸發」應撤回。但風險沒有消失，只是換了形狀，而且更該擔心：**

```
chain_to_stage1.sh:45-48  preflight 只檢查「檔案存在」，不檢查「值對不對」：
    for f in annotations/. procthor_modalities/. sem_edge_cache.json \
             stage1_encoding_protocol.json stage1_hyperparameters.json; do
        [ -e "$OUT/$f" ] || die "missing prerequisite"

我實測 data/outputs/stage1_hyperparameters.json（2026-08-21 產生，現在就存在）：
    learning_rate  0.001    ← Kyzen 08-27 核可的是 5e-4 起跑
    epochs         50       ← Kyzen 08-27 核可的是 250 上限
    weight_decay   0.1      相符
    batch_size     64       相符

chain_to_stage1.sh:51  [ "$ANN" -ge 45000 ] —— n05 跑完那一刻就會過。
```

**後果**：只要 08-28 早上有任何人（含我們任何一個角色）手動起這條鏈，
它會一路 n06 → n09 → Stage 1 smoke 全部跑完，而且：

```
· n09 把「lr 1e-3 / epochs 50」寫死進 stage1_protocol.json 的 hash
· splits.json 生成時沒有 dev-val（第 4 節漏 3 的順序被整個跳過）
· 之後要改超參數就得把 n09 整個重做
```

`:53-55` 有 `pgrep` 擋「別的 metafind stage 在跑」，
但那只擋 GPU 打架，**不擋「用過期的協定生成 artifact」**。

> **⚠ 在 Kyzen 對超參數那批打 `✅` 之前，不得有任何人起 `chain_to_stage1.sh`。**
> 這不是我能決定的事，是要上呈的事。已上呈。

### 🔴🔴 但它連過期值都輪不到 —— `:50` 會回 0，鏈會當場 die

**出處**：ULIP2 REVIEWER `metafindv1-c4`
**我的查證狀態**：**OBSERVED DATA，我剛剛親自量的**

```
find    data/outputs/annotations -maxdepth 1 -name '*.json' | wc -l          →      0
find -L data/outputs/annotations -maxdepth 1 -name '*.json' -type f | wc -l  → 31,860
```

**原因**：`data/outputs/annotations` 是一個 symlink →
`/home/kyzen/metafind_out/annotations`（`ls -ld` 顯示 `lrwxrwxrwx`，2026-08-22 建立）。
`find` 沒有 `-L` 就不跟隨 symlink，只把它當成一個檔名處理，而那個名字不符 `*.json`。

`chain_to_stage1.sh:50` 逐字就是**沒有 `-L` 的那一條**：

```
:50   ANN=$(find "$OUT/annotations" -maxdepth 1 -name '*.json' | wc -l)
:52   [ "$ANN" -ge 45000 ] || die "only $ANN annotations; expected ~45,955"
```

→ **`ANN` 會是 0。鏈會 die，訊息是「only 0 annotations」，而語料實際有 46,024 筆健康的。**

**這是一個 die，不是一次污染。** 沒有任何東西被損壞、沒有資料遺失、沒有 GPU 時間被浪費。
**真正的傷害是那個人會下的結論** —— 跑了五天，早上看到「0 annotations」，
每一個理由都指向「這批跑失敗了」。

**而且 `status.sh:50/51` 有完全相同的問題，我也量了：**

```
status.sh:50   count "n03 點雲"  "$(recs "$OUT/pointclouds") 個"
status.sh:51   count "n04 渲染"  "$(recs "$OUT/renders") 個"

實測  data/outputs/renders      no-L: 1   with-L: 46,053   （symlink）
      data/outputs/pointclouds  no-L: 1   with-L: 1        （symlink，內部結構不同）
```

**兩個儀表會同時報同一個假的 0。** 兩個儀表互相印證的假零，比一個有說服力得多。
這是我要放在 Kyzen 面前的組合。

**修法是兩行加一個旗標**（`find -L`），但那是程式碼，屬 Engineer lane，我不碰。

### 證據等級的變化本身值得記下

ULIP2 REVIEWER 第一次提這兩處時，把它們標為 **INFERENCE 不是 OBSERVED** ——
機制在 `renders` 上已證明，但當時兩個目錄都是空的，`no-L` 與 `-L` 都是 0，量不出差別。
**n05 今晚把 31,860 個檔放進其中一個，同一條指令就從「無法測試」變成「可證明是錯的」。**
`:69`（`embeddings`）仍是 **INFERENCE** —— 那個目錄還是空的。

> **這就是第一次就誠實標註的全部價值。** 當時若硬記成 OBSERVED，今晚什麼都不會改變，
> 這次升級也就不帶任何資訊。

---

## 5b. ESSGNN REVIEWER 把我的撤回理由加硬

**出處**：ESSGNN REVIEWER `metafindv1-9a`
**我的查證狀態**：**CONFIRMED**（論文型別簽名我親自讀過）

我原本用「edge 佔第一層輸入 83%」論證 edge 壓過幾何，後來以「未量測」為由撤回。
它指出**撤回是對的，但理由可以更硬**：

那個論證不只是未量測，它**在架構上就推不出來**。
`m_ij = φ_e(h_i, h_j, ||x_i-x_j||², e_ij)` 是 **concat 進 MLP**。
concat 的第一層 Linear 對每個切片有獨立權重塊：
**一個 512 維切片可以被學成全零，一個 1 維切片可以主導輸出。**

→ 正確說法：**寬度是容量，不是影響力。要講影響力就必須量測。**
→ 而且 MetaFind 自己的型別簽名 `f_h: R^(2d+1+e) → R^d`（`2methdology.tex`）
  就把 `d` 與 `e` 分開命名 —— **論文從未要求兩者可比，也沒說誰該壓過誰。**

這條是 **PAPER FACT ＋ 架構事實**，不需量測即成立。

**它同時拒絕讀我送的四份同步文件**，理由正確：
Kyzen 的凍結範圍包含「讀、查、驗」，而我那封同時說「沒有 ✅、凍結未解除」
又說「讀這四份 908 + 2,309 行、對照六條新規則、打我自陳的第 3/5/6 條」——
**那是審查工作，不是讀信。** 它把先前「可以送，我會讀」收回，是對的。
內容它留著，Kyzen 給 `✅` 的那一刻才開始，不需重送。

**這是本日第二次有人抓到我「把一件事講得比它的機制弱一階」。** 記在這裡。

---

## 6. 兩處我自己在 ULIP2 檔案裡看到、但不屬於我的不一致

**出處**：ESSGNN ENGINEER 讀到後回報（它沒碰那些檔）
**我的查證狀態**：UNVERIFIED

```
annotate_run.py:1    docstring 逐字仍寫 "Run Qwen2.5-VL over each asset's 11 views"
annotate_run.py:84   MODEL_ID 實際是 gemma-4-12B-it
                     → docstring 描述了一個沒在跑的模型

annotate_run.py:101  說 bake-off arms 記的是 /mnt/data1/kyzen/models/gemma-4-12B-it
                     現值是 /home/kyzen/metafind_out/gemma-4-12B-it
                     已標 C1 2026-08-24
```

---

## 7. 待 Kyzen 裁決的清單（本檔彙整，含出處）

```
1  ESSGNN pooling 在 resolve_stage2.py 寫什麼            ESSGNN ENGINEER
   sum（已拍的起點）／ mean（現值）／ None（等短跑）
2  n07c 的多樣性驗收門檻                                  ESSGNN ENGINEER
   它提議：樣本數 ≥ 10 的類別，中位數類別內相異率 ≥ 0.7
3  n07c 覆蓋舊 procthor_object_text.json 還是另存新檔      ESSGNN ENGINEER
4  U-20 要不要排在 n08 之前                               ESSGNN ENGINEER + REVIEWER
   n07c 換文字後 n08 要重跑；U-20 未決就跑 n08 = 跑兩次 GPU
5  C_dev_selection 協定的 gallery 是什麼                   ULIP2 ENGINEER
   它推 gallery = dev_val，需裁
6  dev fold 怎麼做                                        ULIP2 ENGINEER
   它主張 K-fold 只買穩定度買不到外推；
   建議改為「把資料量本身當自變數量兩點（例如 60% 與 72%，val 固定）」看斜率
7  F-A 的 phi_e 尾端 Swish 要不要補                        ESSGNN ENGINEER（我已 CONFIRMED）
8  ledger 對 250 的記錄理由要不要改寫                       ULIP2 REVIEWER
   從「上游說 250」改為「USER-APPROVED，上游預設為佐證」
```

---

## 8. 本檔的地位

**這是搶救記錄，不是協定。** 每一條的權威等級以標註為準：
`CONFIRMED` = 我親自開檔驗過；`UNVERIFIED` = 原樣保存同事的回報，尚未複驗。

**沒有任何一條因為寫進本檔而變成已決策。**
Rule 16 未變：進協定只有三條路 —— MetaFind PAPER FACT ／ Kyzen 明確核可 ／
既有 ledger 條目載明該具體參數。

全部角色於本日 **零寫入、零 commit、零 GPU**。n05（PID 113455）未受影響。

---

## 9. 🔴 儀表全部在說謊 —— 同四行輸出裡四個獨立的失真源

**出處**：ULIP2 ENGINEER `metafindv1-be` ＋ ULIP2 REVIEWER `metafindv1-c4`
**我的查證狀態**：**CONFIRMED**，逐條開檔驗過

`tools/status.sh` 這四行，2026-08-27 05:25 的實際輸出：

```
跑中   n05 標註   45,553 annotated this run, 45,955 complete on disk, 3 quarantined
n05 標註(v3)      0 / 45,955

真實狀態  n05_full.log 尾巴：[ 32000/46024] 9.5/min, quarantine 226
```

**四個不同機制，沒有一個是 symlink：**

```
① :56  v3_count 濾 prompt_version == 3，實測 sidecar 是 8 → 分子恆為 0
       與 symlink 無關。Python 的 glob 會穿透 symlink，-L 蓋不到這個。
② :63  分母寫死 "45,955"，現行目標是 46,024
       隔離數再降就會印出超過 100%
③ :22  tail 的是 n05_v3_full.log（mtime 08-20 07:09），實際在寫 n05_full.log
④ :32  pgrep 找已改名的 chain_after_n05，分支恆真
外加   chain_to_stage1.sh:55 的 die 訊息也寫死 "~45,955"
```

**最陰的是 45,955 出現兩次（② 與 ③），兩行互相佐證，反而更像真的。**

REVIEWER 的判定，我原話保留：

> **假的 0 只是讓人懷疑健康的語料；假的 45,955 讓人相信一個階段做完了 ——
> 而它邀請的動作正好是「開始下一階段」。**

而下一階段正是 `chain_to_stage1.sh`，門檻 45,000。**假訊號剛好邀請被禁止的動作。**

---

## 10. 今日第三次：指令看不到被問的東西，而它乾淨的答案讀起來像世界的事實

三次，三個不同的人，三種不同的機制：

```
1  `| head` 截斷   —— 我加的視窗，被讀成「資料裡從來沒有」
2  `ps` 的比對     —— 早上那次
3  裸 `find <symlink>` 沒有 -name —— 我今晚報「pointclouds 結構不同」，
   並據此把 ENGINEER 派去查一個不存在的差異。
   REVIEWER 重現了我的數字：`find data/outputs/pointclouds -maxdepth 1 | wc -l` → 1，
   那個 1 是 **symlink 自己那一筆**。加上 `-name '*.npz'` 之後：no-L 0 / with-L 46,052。
   四個目錄形狀完全一樣。**是我的指令錯，不是磁碟。已撤回並通知 ENGINEER 停止。**
```

**共同形狀**：指令的作用域被自己縮小了，而輸出沒有任何地方顯示這件事。
`status.sh` 的四個失真源是同一族的第四到第七例 —— 只是它們住在儀表裡，不是住在一次性的查詢裡。

> 這一族不會因為「知道有這回事」而消失。它每次都以一個乾淨、合理、
> 看起來就是答案的數字出現。**唯一擋得住的是「換一個角度再量一次」。**

---

## 11. 2026-08-27 Kyzen 放行紀錄

**原話：「1234」。`✅` 打在 MASTER 視窗，涵蓋四項，僅此一次，不順延。**

```
1  commit 兩處 -L（chain_to_stage1.sh:50 · status.sh:44 recs()）
2  修 chain_to_stage1.sh:72 的 embeddings 同型缺陷
3  status.sh 四個失真源一次修完，含 chain_to_stage1.sh:55 的寫死舊值
4  改 resolve_stage1.py:239 lr 1e-3 → 5e-4 · :243 epochs 50 → 250
   並處理 ENGINEER 的漏 2（五欄進 REQUIRED_HYPERPARAMETERS）
   與漏 3（hash 連動順序 resolve_stage1 → n05b → n09）
```

**這個 `✅` 不涵蓋任何執行。** 不准跑 n09、不准起 chain、不准跑 n05b、不准碰 GPU、
不准順手修其他 open findings、不准加 preflight 的 provenance 守門（列下一輪）。

**記錄紀律（Rule 17，寫死，日後不得被改述）：**

```
· epochs 250 是上限，先跑 5 → 10 → 25 pilot，不是無條件跑滿
· lr 5e-4 是起跑點，第一輪 sweep 2.5e-4 / 5e-4 / 7.5e-4 / 1e-3
· lr_start 1e-6 / lr_end 1e-5 是**數值不是機制**，Kyzen 明確選「照 ULIP」，
  記 USER-APPROVED，不是自動繼承
· ULIP 的 --lr 預設是 3e-3，Kyzen 刻意排除。
  **日後不得把這個 sweep 描述為「照 ULIP 的設定」** —— 照的是機制，偏離的是數值
· 全部記 USER-APPROVED，上游列為佐證而非依據
  （否則後來的人套 Type D 會把 250 重開，因為 --epochs default 正是 Type D）
```

**流程未變**：改完 → 送審（Codex 額度 8:18 AM 恢復，之前依 DL-029 由 REVIEWER 代審）
→ 八項報告 → MASTER 轉呈 → Kyzen 另外放行才執行。
