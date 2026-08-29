# 角色名冊

**更新時機**：Kyzen 重開 session 之後，他會叫 MASTER 點名。不是定期，不是自動。

## 2026-08-29 23:40（第四次點名，六位全到）

| 角色 | session | ref | socket |
|---|---|---|---|
| MASTER | `metafindv1-d7` | `c920c7` | — |
| ULIP2 Block Engineer | `metafindv1-2c` | `943a73` | `/run/user/1002/cc-socks/7037.sock` |
| ULIP2 Block Reviewer | `metafindv1-16` | `869408` ⚠ | `/run/user/1002/cc-socks/6268.sock` |
| ESSGNN Block Engineer | `metafindv1-a9` | `554aa1` | `/run/user/1002/cc-socks/48142.sock` |
| ESSGNN Block Reviewer | `metafindv1-53` | `d57120` | `/run/user/1002/cc-socks/49032.sock` |
| INTEGRATOR | `metafindv1-e7` | `ba9699` | `/run/user/1002/cc-socks/49198.sock` |

## 三個今天真的害過人的陷阱

**一、名字會被回收，角色不跟著走。**
`metafindv1-55` 在兩輪之間從 ULIP2 Block Reviewer 變成 ESSGNN Block Engineer。
ESSGNN Reviewer 連四輪換名（`40` → `a5` → `53`）。
ULIP2 Engineer 曾因此以為自己的 Reviewer 失聯，去找上一輪的名字 —— 人一直在。

**二、🔴 兩個來源必須在同一回合一起重讀。一新一舊會生出一個從未存在過的身分。**
INTEGRATOR 本輪自報 `metafindv1-82 [963894]`，**寄過去被退回「not reachable」**。
它自己查出根因，這條比原本那句「要從環境讀」重要：

> socket 是它那一封現讀的（`49198`，**正確**）；
> name/ref 是更早一次 `ListAgents` 留下來的（`82/963894`，**已過期**）。
> **兩個欄位各自都曾為真，合起來的那個身分從來沒有存在過。**

→ **失效的單位不是單一欄位，是整組身分。**
→ 只強調「要從環境讀」不夠 —— 它照做了，仍然報錯，因為錯的是**另一半**。
→ **兩個來源同時過期才安全；各自新舊才會生出假身分。**

**三、socket 會在同一輪內變，沒有重開也會。**
INTEGRATOR 實測回信途中從 `13562` 變成 `49198`。

**四、兩個來源各缺一半，這是結構性的。**
`ListAgents` 表頭是唯一給得出 **ref** 的來源，**但它給不出自己的 socket**；
`echo $CLAUDE_CODE_MESSAGING_SOCKET` 給得出 **socket**，**卻給不出 ref**。
→ 所以「兩個都讀」不是保險，是**必要**。
→ ⚠ 上表 ULIP2 Block Reviewer 的 `869408` **是 MASTER 從表頭取的，非本人自報** ——
  該 session 本輪讀不到自己的 ref。

## 認領規則

- 沒有角色就回 **UNASSIGNED**。**不要用「這批工作看起來像我做的」反推** ——
  claude-mem 的觀察是專案層級、跨所有 session。
- 兩個來源都讀：`ListAgents` 表頭給 name/ref，`echo $CLAUDE_CODE_MESSAGING_SOCKET` 給 socket。
- 寄不到就**重新點名**，不要用消去法猜。
- 技術議題角色之間直接對接。MASTER 只在 commit 與跨 Block 裁決時進來。
