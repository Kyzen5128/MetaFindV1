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
   Codex（codex:codex-rescue）：只看程式碼本身
```

2026-09-02 前的實況（從對話紀錄數出來的，不是印象）：
8/29–8/30 派工 40 多次、9/1–9/2 凌晨派 7 次稽核；**9/2 上午之後到晚上，0 次**。
十一視角渲染、ESSGNN 改法、λ 初值、資料掃描、清硬碟全是 MASTER 自己寫的，
而且沒送審。這違反 MASTER 的角色卡（「你不做長時間的單一實作」）和三條循環
（改完自動送 REVIEWER＋Codex）。從這份檔案起改回來。

## 二、每一件工作的固定循環（不問、直接跑）

```
1  有疑問  → 先查帳本 / notebook 有沒有討論過 → 沒有才查上游 → 三步都空才問 Kyzen
2  要改碼  → 派給對應 block 的 engineer；MASTER 不自己寫實作
3  改完    → 自動同時送：對應 block 的 reviewer ＋ Codex（共同基準：docs/CODEX_REVIEW_BRIEF_<日期>.md）
4  有發現  → 回 engineer 修 → 再送審，直到 BLOCKER=0 且 MAJOR=0
5  過了    → MASTER commit
6  要開跑  → 只有兩個 block 都審過、Codex 也過，才能開；開跑用 nohup 的 chain script
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

## 四、接下來的流程（十一視角語料）

```
[現在]  四路平行：
        ulip2-engineer   寫 tools/chain_eleven_view.sh ＋ 試跑檢查器 ＋ 清掉舊的第 12 張圖
        ulip2-reviewer   審渲染／Stage 1 的十一視角改動（已 commit 的範圍）
        essgnn-reviewer  審 ESSGNN 池化、MLP 形狀、λ 初值、Stage 2 讀協定
        codex            審整段程式碼
[審過]  MASTER commit → nohup 開 chain → 掛 Monitor
[chain] 0 前檢查（GPU 空、版本 7、11 視角、硬碟夠）
        1 試渲 6 件 → 檢查 sidecar 與相機 → 拼圖放 output/look/eleven_view_pilot/
        2 全語料重渲 46,052 件（估 ~43 小時，3.36 秒/件）→ 檢查數量與隔離率
        3 標註第一階 100 件（gemma，寫到 bakeoff arm，不碰語料）→ 放 output/look/
        停 ← 這裡要 Kyzen 的眼睛（他 8/24 定的：跑完他全審，標記的我修）
[之後]  標註第二階 1,000 → 全語料標註（~80 h）→ n06 編碼（~2–3 h）→ ProcTHOR 同協定
        → query pack → Stage 1 十輪檢查 → 250 輪
```

**順序修正**：早上的計畫把「n06 編碼」排在「重新標註」前面，錯了。n06 看到任何一筆
標註的 `image_identity` 跟渲染不同就整個停（rc 3）；重渲後 45,692 筆全部都會不同。
所以一定是 渲染 → 標註 → 編碼。已改 `workflow/DATA_PLAN_PAPER_FIRST.md`。

## 五、這條 chain 會動到什麼（DL-030 要求的八項）

```
階段        n04 重渲 → n05 第一階
指令        nohup bash tools/chain_eleven_view.sh > data/outputs/logs/chain_eleven_view.log 2>&1 &
寫到哪      data/outputs/renders/<uid>/view_00..10.png ＋ renders/<uid>.json（原地覆蓋）
            data/outputs/bakeoff/eleven_view_r1/（100 筆標註）
            output/look/eleven_view_pilot/、output/look/eleven_view_annot_r1/
多大        約 65 GB（十一張 × 46,052），取代現有 71 GB
多久        試渲 ~2 分；全渲 ~43 小時；標註第一階 ~15 分
覆蓋／刪除  舊的十二視角渲染（版本 6）被原地覆蓋；每個資產目錄裡多出來的 view_11.png 會被刪
            舊十二視角的所有下游產物已在 9/2 搬到 /mnt（DL-081）
錯了能重跑  能。GLB 在 NVMe 與 /mnt 各一份（46,052 件位元組相同）；版本 6 的程式在 git
沒驗證的    相機到底拍出什麼樣，要試渲後用眼睛看（output/look/eleven_view_pilot/）
```
