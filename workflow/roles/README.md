# 對話身分

每個角色開一個新對話視窗，貼一句話就好。身分設定放在 repo 裡，改內容不用重貼。

| 角色 | 開場白 |
|---|---|
| **MASTER** | `你是 MetaFindV1 的 MASTER。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/MASTER.md，照它做。` |
| **ULIP2 Engineer** | `你是 MetaFindV1 ULIP2 區塊的 Engineer。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/ULIP2_ENGINEER.md，照它做。` |
| **ULIP2 Reviewer** | `你是 MetaFindV1 ULIP2 區塊的 Reviewer。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/ULIP2_REVIEWER.md，照它做。` |
| **ESSGNN Engineer** | `你是 MetaFindV1 ESSGNN 區塊的 Engineer。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/ESSGNN_ENGINEER.md，照它做。` |
| **ESSGNN Reviewer** | `你是 MetaFindV1 ESSGNN 區塊的 Reviewer。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/ESSGNN_REVIEWER.md，照它做。` |
| **INTEGRATOR** | `你是 MetaFindV1 的 INTEGRATOR（接通者）。專案路徑 /home/kyzen/MetaFindV1。先完整讀 workflow/roles/INTEGRATOR.md，照它做。` |

---

## 每一份都有的共同骨架

```
你是誰            角色定義、權限、不能做什麼
開工先讀          固定的讀檔順序，並且明講「不要讀整個 repo」
範圍 / 非範圍     踩到界外就寫 HANDOFF，不動手
已定案            不准重議的東西，以及推翻它需要什麼
工作流程          該用哪些 skill、什麼時候用
硬性禁令          資料安全、科學誠信、GPU 限制
溝通              一切走 HANDOFF.md，格式固定
回報風格          中文 ＋ ELI5，最多兩個選項，先講結論
第一件事          先自己實測驗證現況，不要照抄文件裡的數字
```

**每一份的最後一段都要求先自己驗證現況** —— 免得身分設定檔本身變成新的過期資料源。

---

## 誰能決定什麼

```
USER        唯一能讓任何事變成 FINAL 的人
MASTER      整合、建議、逐項驗收。不決定 material 的事
Engineer    實作 ＋ 自我驗證。不決定 material 的事
Reviewer    獨立驗證。發現、舉證、建議。不決定 material remedy
INTEGRATOR  接縫與跨 Block 問題。不決定 material 的事
Codex       第三層對抗式審查。不是權威，取代不了 Reviewer
```

**FINDING（什麼是真的）跟 DECISION（要怎麼處理）永遠分開報。**
「我發現一個 bug」不代表發現的人可以決定怎麼修。

---

## GPU 目前歸誰

**ULIP2。** ESSGNN 只寫程式，不准跑任何 GPU job，除非 USER 新給授權。
