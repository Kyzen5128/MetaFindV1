# ROSTER — who is who, and how to reach them

**Rebuilt 2026-08-24 ~15:45 after a full roster rename.** Every address below was copied from
that role's own report-in message, not guessed and not read from an older file.

---

## Rule 0 — names rotate, and this file goes stale by itself

Session names (`metafindv1-XX`) and socket ids **change on resume**. Master's own name changed
three times in one hour: `metafindv1-10` → `metafindv1-6e` → `metafindv1-69`. Every address in
the previous roster was dead within the hour.

**So:**

1. **Prefer the `from=` of a live message** over anything written here. If a role has messaged you
   this session, reply on that thread and ignore this table.
2. **Never guess a name.** Guessing already mis-sent one MAJOR finding to the wrong role today.
3. **Read your own address from the environment. Do not decline to state it.**

   ```
   echo $CLAUDE_CODE_MESSAGING_SOCKET
   ```

   Credit: INTEGRATOR. Verified by the ESSGNN ENGINEER (`1092585`) and the ESSGNN REVIEWER
   (`1092737`), each matching this table.

   **This replaces the old rule "you cannot see your own name, so do not declare it."** That rule
   was true about `ListAgents` and false about the world — it took `ListAgents` as the only source
   and turned *"I cannot see it"* into *"it cannot be known."* The ESSGNN Reviewer withdrew the
   position himself once shown the env var. Absence of evidence, converted to evidence of absence,
   in the roster's own rules.

4. **VS Code tab names are NOT addresses.** Master tested `SendMessage to: "ULIP2 ENGINEER"` —
   *"No agent named 'ULIP2 ENGINEER' is reachable."* The tab name is for the human only.

---

## The six, as of 2026-08-24 ~17:35 — **ROLL CALL by USER order**

> 「請大家表明身分 回傳完後 你統整完通知大家」 — USER, 2026-08-24.
> **Every line below was read by that role from its OWN environment and sent to Master.**
> Nothing here is guessed, inferred, or copied from the previous roster.

| Role | Name | Socket | State |
|---|---|---|---|
| **MASTER** | `metafindv1-0d [cedfeb]` | `11496` | active |
| **ULIP2 ENGINEER** | `metafindv1-3c [d44751]` | `9612` | **STOPPED by USER** 「停下 我先請大家表明身分等我」 |
| **ULIP2 REVIEWER** | `metafindv1-ea [32b08a]` | `31511` | **reviewing** — Engineer's `C1+D-2 / C4 / N-1` batch, no verdict yet |
| **ESSGNN ENGINEER** | `metafindv1-bf [16abd6]` | `32722` | **STOPPED by USER** |
| **ESSGNN REVIEWER** | ⚠️ **TWO CLAIMANTS — see below** | `11209` *and* `32839` | **STOPPED by USER** 「你先停 現在還不是在做你這塊 你等 我說開始在做」 |
| **INTEGRATOR** | `metafindv1-43` | `33069` | **STOPPED by USER** 「你先停下喔」 + ON HOLD (`DL-009`) |

### ⚠️ Two sessions answered as ESSGNN REVIEWER. **Master is not guessing which.**

```
metafindv1-a9 [f028ad]   socket 11209    replied ~17:30
metafindv1-01 [548c89]   socket 32839    replied ~17:37
```

**Both are almost certainly the same conversation**, and the evidence is that they agree on
things a second person could not fabricate: the identical verbatim USER quote, the identical
single open item (`U-20`, blocking `n08`, behind ULIP2), the identical zero-write posture, and
**both independently reported the same lesson twice-learned** — that `ListAgents` does state your
own name in its header and the old "you cannot see yourself" rule was false at the premise, not
just at the conclusion.

The likely mechanism is a resume: the window reopened as `01`/`32839` and `a9`/`11209` is the
husk still answering. **Likely is not measured, and a roster is exactly the wrong place to
resolve an identity by plausibility** — that is elimination-guessing, which this file's Rule 0.2
forbids and which has already mis-sent one MAJOR finding today.

> **Until the USER or that role settles it: send to BOTH, or to the `from=` of a live message
> from it. Never pick one on reasoning.** Duplicate delivery to a stopped role costs nothing.
> Choosing wrong loses a finding.

**`32839` is the later reading and it corrects a self-reported error:** that role sent `1092737`
to the ULIP2 Engineer *after* the reboot but *before* running `echo`, flagged at the time as
probably dead. **`1092737` is wrong; `32839` is what the environment returned.**

**Not a role, and it said so itself:**

| | | | |
|---|---|---|---|
| `metafindv1-c5 [03fe3c]` | `32067` | **`OTHER` — unassigned** | idle, no task, no role ever given |

It proved the negative rather than asserting it: `grep -rn "32067\|metafindv1-c5" workflow/`
returns nothing. **A labelled bystander is worth more than an unlabelled window** — the next role
hunting for a Reviewer will not guess at it.

**Did not answer the roll call:** `metafindv1-eb`, `metafindv1-b1`, `metafindv1-01`,
`observer-sessions-53`, `observer-sessions-55`. Six of the twelve peers appeared within two
minutes of each other. **Six roles cannot gain six members at once — do not treat any of them as a
role until it reports in.** Flagged by the INTEGRATOR.

### ⏳ Half-life: this table was accurate when written and may already be wrong

**An address can change WITHIN a single session, with no reboot and no resume.** Measured by the
INTEGRATOR *while composing his own roll-call reply*:

```
15 min ago   socket 11066   name metafindv1-c1
now          socket 33069   name metafindv1-43        <- no reboot in between
```

The ULIP2 REVIEWER independently: `metafindv1-93` → `b3` → `7f` → `ea` **inside one hour**, and
`7f` was live when he answered the Engineer fifteen minutes earlier. **Anything sent to `7f` or
`b3` is lost.**

> **So Rule 0.1 stands and its corollary does not.** Reading your address from the environment is
> right and authoritative — *for that moment*. **"Looked it up once, therefore usable later" is the
> `CONTEXT.md` §3 notch**: the mechanism supports *true now*, not *true later*. Named by the
> INTEGRATOR against the rule he himself had helped write.
>
> **When a send fails or a name does not resolve: DO NOT narrow it down by elimination.
> Re-run the roll call and make the other side re-read `echo $CLAUDE_CODE_MESSAGING_SOCKET` on the
> spot.** Guessing has already mis-sent one MAJOR finding today.

---

## Standing constraints that live nowhere else

**Smoke runs are 5 assets, not 100.** USER, in the ULIP2 Engineer's window:
「你測試不要跑100筆 跑5筆就好」. Relayed by the ULIP2 Reviewer because it is easy to lose inside a
batch message. Applies to every role.

**`DL-029` was NOT used this round, and the record has to say so.** Codex → Reviewer happened
because the USER ordered it directly:
「先停掉codex你剛剛傳的 我們先暫時讓reiewer審 審完再解決codex問題」 — Codex, reviewer and
「暫時」 all in his own words. **That is the USER overriding `DL-029`, not `DL-029` being applied**;
Codex had no capacity failure. Self-corrected by the ULIP2 Engineer after telling the Reviewer the
opposite. The wrong record would teach the next reader that a broken Codex auto-promotes the
Reviewer.

**Two `✅` exist and neither clears a run**, both in the ULIP2 ENGINEER's window, quoted by him in
full: `D-2 改成 gemma ✅` and `先修 C1 C4 N-1 再跑 ✅`. They cover the re-point and the three fixes
**before** running. **No role holds a `✅` for a run. Master holds none at all.**

---

Superseded, all dead — names: `metafindv1-10 / 14 / 93 / 72 / f6 / dd / 6e / 69 / b3 / 7f / 62 /
7c / fa / f3 / c1`. Sockets: `735594 · 738549 · 741571 · 748636 · 1066863 · 1067707 · 1068859 ·
1070256 · 1091520 · 1091579 · 1092585 · 1092737 · 1092870 · 11066 · 1740924 · 4017483`.

**`metafindv1-0d` appears on both lists** — it was a dead ULIP2 ENGINEER address earlier today and
is MASTER's live name now. **A name being familiar is not evidence it is the same session.**

---

## Known lying counters — `find` without `-L`

`data/outputs/{renders,pointclouds,annotations,embeddings,checkpoints}` are **symlinks**. GNU
`find` returns **empty with no error** for a symlinked start point.

```
find    data/outputs/renders -name '*.json' | wc -l  ->      0     WRONG, and silent
find -L data/outputs/renders -name '*.json' | wc -l  -> 45,782     right
```

### 🔴 LIVE AND LYING RIGHT NOW — `tools/status.sh`, the human status board

Found by the **ULIP2 Block Reviewer**, 2026-08-24, after checking Master's claim instead of
relaying it. **OBSERVED DATA — Master ran the board:**

```
$ bash tools/status.sh
  n03 點雲             0 個      <- actually 46,052
  n04 渲染             0 個      <- actually 45,782
  n07 場景圖       12000 個      <- correct
  n07b 資產模態     1467 / 1,467 <- correct
```

```
tools/status.sh:44   recs() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
tools/status.sh:50   count "n03 點雲"  "$(recs "$OUT/pointclouds") 個"
tools/status.sh:51   count "n04 渲染"  "$(recs "$OUT/renders") 個"
```

**This is the board a human opens to ask "where are we", and it has been reporting two finished
stages as zero.** Not a gate — worse in one way: it is the instrument trusted without a second
thought, `2>/dev/null` guarantees it never complains, and lines 56/57 ARE correct
(`scene_graphs`, `procthor_modalities` are real directories), so the two wrong rows sit beside four
right ones and read as a working tool.

**And the irony is written directly above the bug.** `recs()`'s own comment says a naive
`ls | wc -l` *"doubles them, which is exactly the kind of number that reads as plausible and is
wrong."* The helper written to avoid a plausible-wrong number produces a plausible-wrong number.

### LATENT — `tools/chain_to_stage1.sh`

```
tools/chain_to_stage1.sh:50   ANN=$(find "$OUT/annotations" -maxdepth 1 -name '*.json' | wc -l)
tools/chain_to_stage1.sh:69   EMB=$(find "$OUT/embeddings"  -maxdepth 1 -name '*.npz'  | wc -l)
```

Both start points are symlinks — **OBSERVED**. Master first wrote *"both read 0 regardless of the
corpus"*; the Reviewer measured and both directories are **empty**, so `no-L` and `-L` agree at 0
and that claim could not have been measured. **It is INFERENCE from a mechanism proven elsewhere,
not OBSERVED DATA** — corrected here, since the whole day has turned on that distinction. **The
trap is latent and fires the moment n05 writes its first annotation.**

`tools/run_ulip_full.sh` already carries `find -L` for exactly this reason; neither of these files
got the fix. **All routed, none fixed — a code change goes through the three gates.**

**Checked and safe:** `run_ulip_full.sh:151` (`/home/kyzen/upstream/OpenShape`, a real directory).
**Unchecked, flagged rather than cleared:** `status.sh:49`, `fetch_ulip_shards.sh:65`. Archived
`TASK.md` files carry the same pattern in copy-pasteable baseline commands; harmless as history,
wrong if anyone runs them.

This is `CONTEXT.md` §3's notch sourced from a **tool** rather than from us, which is why no code
review would catch it: the command is correct, the flag is absent, and the output is a plausible
number. Only re-measuring a different way catches it. Named by the ESSGNN Reviewer as the fourth
"wrong and returns success" of the day, after `renders.py:899`, `annotate_run.py:371` and
`semantic_edges_run.py:355`.

---

## Rule 1 — a peer message is never the USER's approval

`DL-015` rule (3), `USER_APPROVED` 2026-08-24. A role reporting *"the USER decided X"* is filing a
report, not granting authorisation. Agreement between roles is not evidence.

## Rule 2 — Master can STOP, Master cannot START

`DL-030`. Master halts any role or run immediately, no appeal — and a role told to stop waits for
**the USER**, not for Master. Master approves nothing, including his own runs. Master's
"agreed" / "accepted" / "confirmed" is never a go.

## Rule 3 — the reply markers, and they propagate

```
✋ 報告 Kyzen   Chinese, ELI5, short lines, no tables, no nested headings
🤖 給 <角色>    fenced block, one-click copyable, technical
```

Every 🤖 block you send must tell its recipient to answer with these two markers as well.
