# 角色名冊

**更新時機**：Kyzen 重開 session 之後，他會叫 MASTER 點名。**不是定期，不是自動。**
位址只在重開後才變。

## 2026-08-29 09:50（第二次點名）

| 角色 | session | ref | socket |
|---|---|---|---|
| MASTER | `metafindv1-0c` | `ac1f75` | — |
| ULIP2 Block Engineer | `metafindv1-0c` | `9e55e6` | `/run/user/1002/cc-socks/6413.sock` |
| ULIP2 Block Reviewer | `metafindv1-55` | `bf024c` | `/run/user/1002/cc-socks/7617.sock` |
| ESSGNN Block Engineer | `metafindv1-a1` | `712b25` | `/run/user/1002/cc-socks/8791.sock` |
| ESSGNN Block Reviewer | `metafindv1-40` | `71d471` | `/run/user/1002/cc-socks/18218.sock` |
| INTEGRATOR | `metafindv1-60` | `dc6c34` | `/run/user/1002/cc-socks/17807.sock` |

回報 UNASSIGNED：`metafindv1-73 [1053db]` · `metafindv1-c4 [79e643]`
未回報 / 非角色：`metafindv1-e9` · `metafindv1-45` · `observer-sessions-*`

## 🔴 兩個必須帶 ref 的理由，兩個今天都真的害過人

**一、MASTER 與 ULIP2 Block Engineer 這一輪同名，都是 `metafindv1-0c`。**
只差 ref（`ac1f75` / `9e55e6`）。寄錯會寄回自己。

**二、名字會被回收，角色不跟著走。**
`metafindv1-77` 在 08-29 的兩輪之間，從 ESSGNN Block Engineer 變成 ULIP2 Block Reviewer。
ULIP2 Block Engineer 因此以為自己的 Reviewer 失聯，去找上一輪的 `metafindv1-c4` ——
那個名字這一輪是 UNASSIGNED。**人一直在，只是他在找舊位址。**

## 認領規則

- 沒有角色就回 **UNASSIGNED**。**不要用「這批工作看起來像我做的」反推。**
  claude-mem 的觀察是專案層級、跨所有 session，不能當成「我做過」的證據。
- 兩個來源都要讀：`ListAgents` 表頭給 name/ref，`echo $CLAUDE_CODE_MESSAGING_SOCKET` 給 socket。
- 寄不到時**重新點名**，不要用消去法猜。
