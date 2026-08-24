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

## The six, as of 2026-08-24 15:45

| Role | Name | Socket | State |
|---|---|---|---|
| **MASTER** | `metafindv1-69 [349be2]` | — | active |
| **ULIP2 ENGINEER** | `metafindv1-62` | `1091579` | active — n04 halted 45,782/46,052. **Stopped by Kyzen.** |
| **ULIP2 REVIEWER** | `metafindv1-b3 [5eb15e]` | `1091520` | active — see the review-state note below |
| **ESSGNN ENGINEER** | `metafindv1-7c [59f98b]` | `1092585` | **FULLY STOPPED** |
| **ESSGNN REVIEWER** | `metafindv1-fa` | `1092737` | **FULLY STOPPED** |
| **INTEGRATOR** | `metafindv1-f3` | `1092870` | **ON HOLD** (`DL-009`) |

The ULIP2 Engineer's own message says its session is `metafindv1-0d`; the delivered `from-name`
was `metafindv1-62`. **The `from` wins** — a session cannot see its own listed name, which is
exactly what Rule 0.3 exists for.

> ### ⚠️ **A verdict without its subject.** Corrected 2026-08-24, at the ULIP2 Engineer's request.
>
> This row first read *"`DL-029` review of the batch returned PASS on the CODE"*. **Beside the
> Engineer's `n04 halted 45,782` it reads as "reviewed and ready to run". It is not.**
>
> ```
> REVIEWED, round 2 PASS   the 10-file batch: n06's view_io bypass, the cache-generation
>                          binding, rebuild_index, the fingerprint width, the failure
>                          classifier, the 3 blockers introduced with the circuit break.
>                          -> this is the code that produced the 45,782 on disk.
>
> NOT REVIEWED BY ANYONE   the guard rewrite now in the working tree:
>                          blankness  std(black-composite) -> alpha, MIN_COVERAGE 0.001
>                          distinct   all-12-byte-distinct -> only all-12-identical refused
>                          breaker    DETERMINISTIC_INPUT no longer counts toward SYSTEMIC_RUN
>                          new fields view_coverage / distinct_views / dark_views
>                          effect     201 of 270 recovered, 99.41% -> 99.85%
>                          -> no request sent, no round 1, no verdict.
> ```
>
> **Reviewed on the code that produced the corpus; unreviewed on the code that would change which
> assets count.** A verdict detached from its subject is the same defect as a count detached from
> its denominator — `CONTEXT.md` §3. Caught by the ULIP2 Engineer against a row that favoured him.
>
> And it would clear no run either way: `DL-030` puts Kyzen last, and a peer relaying a verdict is
> a report, not an authorisation (`DL-015` rule 3).

Superseded, all dead: `metafindv1-10 / 14 / 93 / 72 / f6 / dd / 0d`, sockets
`735594 · 738549 · 741571 · 748636 · 1066863 · 1067707 · 1068859 · 1070256 · 1740924 · 4017483`.

---

## Known lying counters — `find` without `-L`

`data/outputs/{renders,pointclouds,annotations,embeddings,checkpoints}` are **symlinks**. GNU
`find` returns **empty with no error** for a symlinked start point.

```
find    data/outputs/renders -name '*.json' | wc -l  ->      0     WRONG, and silent
find -L data/outputs/renders -name '*.json' | wc -l  -> 45,782     right
```

**Live, unfixed, found 2026-08-24 by scanning for it:**

```
tools/chain_to_stage1.sh:50   ANN=$(find "$OUT/annotations" -maxdepth 1 -name '*.json' | wc -l)
tools/chain_to_stage1.sh:69   EMB=$(find "$OUT/embeddings"  -maxdepth 1 -name '*.npz'  | wc -l)
```

Both start points are symlinks. **Both counters read 0 regardless of the corpus**, and whatever
gate reads them sees an empty stage. `tools/run_ulip_full.sh` already carries `find -L` for this
exact reason; `chain_to_stage1.sh` never got the fix. **Routed, not fixed — changing it is a code
change and goes through the three gates.** Archived `TASK.md` files carry the same pattern in
copy-pasteable baseline commands; harmless as history, wrong if anyone runs them.

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
