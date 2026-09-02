# 自動接力設定（2026-09-02 起生效）

Kyzen 2026-09-02：「我覺得你最近寫程式都感覺很被動 監控也是 可以解決嗎? 做完自動下一步」
「設定好跟我報備 目前設定是? 以及接下來流程先設定好」。
這份檔案回答這兩句。狀態以 repo 現況為準；這裡只講規則和流程。

## 一、角色（Kyzen 設計的樹，`.claude/agents/`）

```
MetaFind MASTER（這個視窗）
   ├─ ulip2-engineer ──→ ulip2-reviewer
   ├─ essgnn-engineer ─→ essgnn-reviewer
   └─ integrator（兩個 block 都審過之後才進場）
```

**Codex 不在這棵樹裡。**（Kyzen 2026-09-02 晚：「不要有codex審查這個機制 拔掉 然後codex
審查由我呼叫 你們要審查互相呼叫 reviewer就好」）Codex 只有 Kyzen 自己會叫。MASTER、
engineer、reviewer 都不准派 Codex，也不准把「Codex 過了」寫進任何開跑條件。
需要第二意見時，找**另一個 block 的 reviewer**：ULIP2 的東西可以請 ESSGNN Reviewer
反過來看，反之亦然。這取代原本的「reviewer ＋ Codex 雙軌」。

2026-09-02 前的實況（從對話紀錄數出來的，不是印象）：
8/29–8/30 派工 40 多次、9/1–9/2 凌晨派 7 次稽核；**9/2 上午之後到晚上，0 次**。
十一視角渲染、ESSGNN 改法、λ 初值、資料掃描、清硬碟全是 MASTER 自己寫的，
而且沒送審。這違反 MASTER 的角色卡（「你不做長時間的單一實作」）和三條循環
（改完自動送 REVIEWER）。從這份檔案起改回來。

## 二、每一件工作的固定循環（不問、直接跑）

```
1  有疑問  → 先查帳本 / notebook 有沒有討論過 → 沒有才查上游 → 三步都空才問 Kyzen
2  要改碼  → 派給對應 block 的 engineer；MASTER 不自己寫實作
3  改完    → 自動送對應 block 的 reviewer（共同基準：docs/REVIEW_BRIEF_<日期>.md）
           跨 block 或想要第二雙眼睛 → 再送另一個 block 的 reviewer。不送 Codex。
4  有發現  → 回 engineer 修 → 再送審，直到 BLOCKER=0 且 MAJOR=0
5  過了    → MASTER commit
6  要開跑  → 只有兩個 block 的 reviewer 都審過才能開；開跑用 nohup 的 chain script
7  開跑後  → 立刻掛 Monitor（persistent）盯 chain log；每個階段轉換都會叫醒 MASTER
8  叫醒後  → 讀結果 → 過關就讓 chain 繼續、不過就停下來修 → 到「要 Kyzen 眼睛」的關卡才通知他
```

只有這些才找 Kyzen：超參數／訓練配方的**數值**、架構選擇、評估協定與 Table 1/2/3、
要用他眼睛看的東西（渲染圖、標註品質）、以及會改變科學結論的重跑。

## 三、監控的機制（為什麼以前會「停掉」）

`nohup … &` 跑完**不會**叫醒 MASTER；等待迴圈超時的通知跟跑完長得一樣，所以以前讀錯四次。
現在：

- 長工作一律 `nohup bash tools/chain_*.sh > logs/… &`（不會因為視窗關掉而死）
- 同一回合馬上 `Monitor(persistent=true)` 跟著 log 走，濾出
  `=== 階段 ===`、`OK`、`STOPPED`、`Traceback`、`quarantine`、`rendered this run`
- 每一行事件都會叫醒 MASTER；MASTER 讀完就做下一步，不等人催
- 到人工關卡：`PushNotification` 通知 Kyzen 一句話，並把要看的東西放 `output/look/`

## 四、目前的流程（2026-09-03 起，依 `REPRODUCTION_PROTOCOL_20260903.md`）

**十一視角重渲已取消排程。** 規格 問題 3 判定：相機協定 UNRESOLVED，現在重渲等於把猜測
固化，若之後猜錯要全部重跑。所以改成**用現有的 12 視角語料把系統做對**。
§四 到 §五 原本那條重渲 chain 作廢，不要照它跑。

```
PHASE 1  現況盤點（唯讀）                         ✅ 完成 workflow/PHASE1_AUDIT_20260903.md
         四十題全部有答案，三十二題由三位稽核者實測

PHASE 2  資料清單 / provenance                    ✅ 完成 tools/build_dataset_manifest.py
         一個 UID 一列，圖片與描述當子紀錄，一個特徵都沒重算

PHASE 3  可逆前處理                               ✅ 本來就完成
         12 張逐視角特徵早就存在每個 embeddings/*.npz 的 `views (12,1280)` 裡

PHASE 4  Dataset API                              ✅ 完成 metafind/data/observation.py
         positive_policy 同 UID ＋ 逐模態的 query / gallery 觀測政策
         gallery_test / gallery_full 本來就支援

PHASE 5  Stage 1 正確性測試                       ✅ 完成
         PointBERT 收到梯度、文字影像凍結、loss 單向、遮罩 30% 獨立、
         全遮 2.7%、遮罩非補零 —— 全部實測過（PHASE1_AUDIT B 段）
         2026-09-03 01:47 在 GPU 上跑完 128 件 1 輪，code_revision 5f64023、
         code_dirty False。訊號與審查員事前預測完全一致：
         train_stage1 0 列（2 步，step%20 不會觸發）、dev_val 1 列
         （0 列才是真的壞掉）。R@1 0.9821，gallery 128 —— 這是接線檢查，不是成績。
         正式 checkpoint 沒被動到（.smoke128 後綴）

PHASE 6  評估敏感度   ← 下一步，需要 Kyzen 放行
PHASE 7  決定哪些協定值得重訓
PHASE 8  才決定昂貴的資料重製（重渲、全量標註）
```

## 五、目前擋在哪

**等 reviewer 放行 GPU。** 規則是兩個 block 的 reviewer 審過才能開跑；
ULIP2 Reviewer 第一輪回 CHANGES REQUIRED（BLOCKER 2），已全部修好並送二審。

**四件事等 Kyzen 裁，都不擋現在的實作：**

```
一  λ 初值 9.0 與 ESSGNN pooling —— 兩個權威衝突，MASTER 不選
    （帳本 DL-077 你裁 0.1；新規格說兩者 UNRESOLVED 且禁止自行決定）
    protocol 上那句假的歸屬已改掉，數值沒動
二  253 個渲染壞掉的已收錄資產（11 個近乎全黑，47 個在封存測試集）
    現在標記起來留著，因為規格 §六 要的就是「不得靜默丟棄」
三  遮罩向量在權重衰減組裡，訓練完與初始化無法區分，只有真實向量的 2%
四  ProcTHOR 節點文字要不要修（工具寫好了，dry run 過了，沒跑）
```

## 六、這一輪動到什麼

```
改了程式        metafind/train/stage1.py       視角數改成讀協定
                metafind/data/splits.py         排除名單真的讀 uid ＋ 過濾階梯
                metafind/models/resolve_stage1.py  完整的 view_aggregation 區塊
                metafind/data/encode_text_image.py 整個區塊對照程式重新驗證
                metafind/eval/run_retrieval.py  QueryPack 少一個參數（reviewer 抓到）
                metafind/data/observation.py    新，觀測政策
新工具          tools/build_dataset_manifest.py
                tools/repair_procthor_node_text.py（不跑）
新產物          data/outputs/manifest/          十個檔，重算 0 個特徵
改了協定        stage1_encoding_protocol.json   多了 view_aggregation
                stage2/essgnn protocol          只改 decided_by，數值沒動
沒有動          渲染、標註、點雲、checkpoint、任何 embedding
測試            1,023 通過
```
