#!/usr/bin/env python3
"""Structural checker for docs/graph/.

Every check here exists because it caught a real bug. Running it is the only way
the six specification documents stay consistent with each other; reading them
does not scale.

    python3 tools/check_graph.py

Exit 0 = all checks pass, 1 = at least one failure.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parents[1] / "docs" / "graph"

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def load(fname: str) -> dict:
    return yaml.safe_load((DOCS / fname).read_text())


def text(fname: str) -> str:
    return (DOCS / fname).read_text()


# --------------------------------------------------------------------------
spec = load("graph_spec.yaml")
registry = load("node_registry.yaml")
plan = load("validation_plan.yaml")

channels = {c["name"]: c for c in spec["state"]["channels"]}
nodes = {n["id"]: n for n in registry["nodes"]}
subgraph_nodes = {
    n["id"]: n
    for sg in registry.get("subgraph_nodes", {}).values()
    for n in sg.get("nodes", [])
}
all_nodes = {**nodes, **subgraph_nodes}
edges = spec["edges"]

WILDCARD = {"ALL_NODES"}
PSEUDO = {"RA-1", "RA-2", "RA-3", "RA-4"}  # audits write records but are not graph nodes
TERMINALS = {"HALT_FAILED", "HALT_BLOCKED", "HALT_INVALIDATED"}


def is_subgraph(nid: str) -> bool:
    """Subgraph-internal nodes are declared without reads/writes on purpose."""
    return nid.startswith("sg")



# --- 1. edge endpoints exist ----------------------------------------------
for e in edges:
    for side in ("from", "to"):
        check(
            f"edge {e['id']} {side}",
            e[side] in all_nodes or e[side] in TERMINALS,
            f"{e[side]} is not a declared node",
        )

# --- 2. edges carry only declared channels --------------------------------
for e in edges:
    for ch in e.get("carries", []):
        check(f"edge {e['id']} carries", ch in channels, f"unknown channel {ch}")


# --- 3. acyclicity of the main line ---------------------------------------
main_edges = [e for e in edges if e["from"] in nodes and e["to"] in nodes]
adj = defaultdict(list)
for e in main_edges:
    if e.get("kind") != "feedback":
        adj[e["from"]].append(e["to"])

WHITE, GREY, BLACK = 0, 1, 2
colour = defaultdict(int)


def visit(u: str, stack: list[str]) -> None:
    colour[u] = GREY
    for v in adj[u]:
        if colour[v] == GREY:
            FAILURES.append(f"cycle in main line: {' -> '.join(stack + [u, v])}")
        elif colour[v] == WHITE:
            visit(v, stack + [u])
    colour[u] = BLACK


for n in nodes:
    if colour[n] == WHITE:
        visit(n, [])
CHECKS += 1


# --- 4. reachability -------------------------------------------------------
targets = {e["to"] for e in main_edges}
sources = {e["from"] for e in main_edges}
roots = sources - targets
reach: set[str] = set()
frontier = list(roots)
while frontier:
    u = frontier.pop()
    if u in reach:
        continue
    reach.add(u)
    frontier.extend(adj[u])
for n in nodes:
    check(f"reachable {n}", n in reach or n in roots, "not reachable from any root")


# --- 5. channel provenance: declared writers == nodes that write it --------
# This caught a stale renamed channel and ten write mismatches.
actual_writers = defaultdict(set)
actual_readers = defaultdict(set)
for nid, n in all_nodes.items():
    for ch in n.get("writes", []) or []:
        actual_writers[ch].add(nid)
    for ch in n.get("reads", []) or []:
        actual_readers[ch].add(nid)

for name, c in channels.items():
    declared_w = set(c.get("writers", []))
    if declared_w & WILDCARD:
        continue
    got = {n for n in actual_writers[name] if not is_subgraph(n)} - PSEUDO
    want = {n for n in declared_w if not is_subgraph(n)} - PSEUDO
    check(
        f"writers({name})",
        got == want,
        f"channel declares {sorted(want)} but nodes write {sorted(got)}",
    )

# --- 6. reader symmetry ----------------------------------------------------
# Same failure mode as writers; caught eleven mismatches on its first run.
for name, c in channels.items():
    declared_r = set(c.get("readers", []))
    if declared_r & WILDCARD:
        continue
    got = {n for n in actual_readers[name] if not is_subgraph(n)}
    want_r = {n for n in declared_r if not is_subgraph(n)}
    check(
        f"readers({name})",
        got == want_r,
        f"channel declares {sorted(want_r)} but nodes read {sorted(got)}",
    )

# --- 7. no node reads or writes an undeclared channel ---------------------
for nid, n in all_nodes.items():
    for ch in (n.get("reads", []) or []) + (n.get("writes", []) or []):
        check(f"{nid} channel", ch in channels, f"unknown channel {ch}")


# --- 8. gate criteria may only invoke channels the gate reads -------------
# G1 once claimed to check ProcTHOR while reading no ProcTHOR channel, so a
# missing dataset would have passed. A criterion that names a channel the gate
# cannot see is decoration.
for g in plan["level_3_gates"]:
    gid = g["gate_id"]
    if gid not in nodes:
        FAILURES.append(f"gate {gid} in validation_plan is not a node")
        CHECKS += 1
        continue
    can_see = set(nodes[gid].get("reads", []) or [])
    blob = " ".join(str(v) for v in g.values())
    for name in channels:
        if "_" not in name:
            continue  # single English words like `splits` produce false hits
        if re.search(rf"\b{re.escape(name)}\b", blob) and name not in can_see:
            FAILURES.append(
                f"gate {gid} criterion names `{name}` but does not read it"
            )
        CHECKS += 1


# --- 7c. the implementation-status table must match the filesystem ---------
# docs/graph/README.md claims "28 non-gate nodes, 2 implemented". A status
# table is exactly the kind of claim that rots silently: it is written once,
# read as current forever, and nothing recomputes it. This is also the claim a
# reader is most likely to mistake specification completeness for.
readme_txt = (DOCS / "README.md").read_text()
non_gate = [n for n in registry["nodes"] if not n["id"].startswith("G")]
# Both numbers come from the heading; hardcoding either one here meant that
# adding a node broke the check with a NameError instead of a message saying
# the count had moved.
m = re.search(r"(\d+) 個非 gate 節點裡，\*\*有程式的是 (\d+) 個\*\*", readme_txt)
check("README implementation-status heading present", m is not None,
      "the status table's heading changed shape; update this check with it")
if m:
    src = "\n".join(
        p.read_text()
        for d in ("metafind", "tools", "setup")
        for p in (DOCS.parents[1] / d).rglob("*")
        if p.suffix in (".py", ".sh")
    )
    # An explicit marker, not a substring hit: a comment that merely NAMES a
    # node id is not an implementation of it, and the first version of this
    # check counted three such comments as working code.
    implemented = re.findall(r"^# IMPLEMENTS-NODE: (\S+)$", src, re.M)
    check("README non-gate node count", len(non_gate) == int(m.group(1)),
          f"table says {m.group(1)}, registry has {len(non_gate)}")
    check("README implemented-node count", len(implemented) == int(m.group(2)),
          f"table says {m.group(2)}, filesystem shows {len(implemented)}: {sorted(implemented)}")

_ru = spec["risks_unknowns"]
_ru_items = _ru["unknowns"] if isinstance(_ru, dict) else _ru
_u_entries = [u for u in _ru_items
              if isinstance(u, dict) and str(u.get("id", "")).startswith("U-")]
registered_ids = {u["id"] for u in _u_entries}
# `marked` is what separates these. Counting every U- id as UNKNOWN is what let
# U-34 sit in the registry as RESOLVED while three documents still called it
# open: the checker had no way to tell the two apart, so the contradiction was
# invisible to it.
unknown_ids = {u["id"] for u in _u_entries if u.get("marked") == "UNKNOWN"}
resolved_ids = {u["id"] for u in _u_entries if u.get("marked") == "RESOLVED"}

m3 = re.search(r"U registry 共 (\d+) 項，其中 (\d+) 項 unresolved、(\d+) 項 resolved", readme_txt)
check("README UNKNOWN tally present", m3 is not None,
      "the UNKNOWN tally changed shape; update this check with it")
if m3:
    total, unres, res = (int(m3.group(i)) for i in (1, 2, 3))
    check("README UNKNOWN total", total == len(registered_ids),
          f"README says {total}, registry has {len(registered_ids)}")
    check("README unresolved count", unres == len(unknown_ids),
          f"README says {unres}, registry has {len(unknown_ids)}")
    check("README resolved count", res == len(resolved_ids),
          f"README says {res}, registry has {len(resolved_ids)}: {sorted(resolved_ids)}")

# A RESOLVED entry must carry its provenance. Without this an entry can be
# flipped to RESOLVED with no decision, no decider and no evidence, and every
# downstream document that repeats the decision has nothing to cite -- which is
# exactly the state that made U-34's half-migration invisible.
for u in _u_entries:
    if u.get("marked") != "RESOLVED":
        continue
    for field in ("decided", "decided_by", "decided_at", "decision_basis",
                  "confidence"):
        check(f"{u['id']} RESOLVED carries {field}",
              bool(str(u.get(field, "")).strip()),
              f"{u['id']} is marked RESOLVED but has no {field}")
    check(f"{u['id']} confidence vocabulary",
          u.get("confidence") in ("low", "moderate", "high"),
          f"{u['id']} confidence is {u.get('confidence')!r}, not low|moderate|high")

# The machine spec itself must not still describe a resolved item as open.
# This is the check the U-08a migration needed and did not have: the registry
# said RESOLVED while `stage2_pairing`'s type was still
# `{gallery_uid, method, confidence}` -- a schema encoding the very
# ProcTHOR-to-Objaverse mapping the decision had removed. 1,949 structural
# checks passed, because a channel's TYPE agreeing with its own readers says
# nothing about whether it agrees with a decision recorded elsewhere.
# Three shapes, all seen in this repo: "[UNKNOWN U-nn]", "U-nn -- BLOCKING",
# and prose that keeps a resolved reading alive as a live option ("Reading A
# stays an open candidate for U-21", "BOTH READINGS REMAIN EPISTEMIC
# CANDIDATES"). The third shape is the one that slipped through: it names no
# bracket and no keyword, and it says exactly the opposite of RESOLVED.
_STALE = re.compile(
    r"(?:UNKNOWN\s+(U-[0-9a-z]+)"
    r"|\b(U-[0-9a-z]+)\s*--\s*(?:BLOCKING|BOTH READINGS)"
    r"|BOTH READINGS REMAIN[^.]*?\b(U-[0-9a-z]+)?"
    r"|open candidate for (U-[0-9a-z]+)"
    r"|(U-[0-9a-z]+)[^.]{0,80}?open candidate)"
)
for name, c in channels.items():
    blob = f"{c.get('type','')} {c.get('note','')}"
    for m in _STALE.finditer(blob):
        uid = next((g for g in m.groups() if g), None)
        if uid is None:
            continue
        check(f"channel {name} does not call {uid} unknown",
              uid not in resolved_ids,
              f"channel `{name}` still marks {uid} as UNKNOWN/BLOCKING, "
              f"but the registry resolved it")
for nid, n in all_nodes.items():
    blob = f"{n.get('notes','')} {n.get('postcondition','')} {n.get('purpose','')}"
    for m in _STALE.finditer(blob):
        uid = next((g for g in m.groups() if g), None)
        if uid is None:
            continue
        check(f"node {nid} does not call {uid} unknown",
              uid not in resolved_ids,
              f"node `{nid}` still marks {uid} as UNKNOWN/BLOCKING, "
              f"but the registry resolved it")
for item in plan["level_1"] + plan["level_2"] + plan["level_3_gates"]:
    blob = " ".join(str(v) for v in item.values())
    label = item.get("id") or item.get("gate_id")
    for m in _STALE.finditer(blob):
        uid = next((g for g in m.groups() if g), None)
        if uid is None:
            continue
        check(f"validation {label} does not call {uid} unknown",
              uid not in resolved_ids,
              f"{label} still marks {uid} as UNKNOWN/BLOCKING, "
              f"but the registry resolved it")


# Prose must not still call a resolved item open. This is the check that would
# have caught the c43c72e half-migration on its own: the registry said RESOLVED
# while three documents still said "取決於 U-34", and nothing compared them.
# "阻斷級" is on this list because it describes an unknown that is CURRENTLY
# stopping the pipeline. Once the registry resolves it, a document still calling
# it blocking is telling a reader to wait for a decision that has been made --
# the same half-migration as U-34, in a different vocabulary.
_OPEN_PHRASES = ("未解", "尚未確立", "取決於", "仍待", "阻斷級",
                 "unresolved", "still open", "open candidate",
                 "BOTH READINGS REMAIN", "epistemic candidate")
# A correction log RECORDS that something was once open. Rewriting it to match
# today's state would destroy the only account of how the decision was reached,
# so everything from the log heading onward is exempt -- history is supposed to
# disagree with the present.
_HISTORY_HEADING = "## 16. 修正紀錄"
for doc in ("README.md", "01_GRAPH_SPEC.md", "02_BUILD_STEPS.md", "00_FINDINGS.md"):
    body = (DOCS / doc).read_text()
    live = body.split(_HISTORY_HEADING)[0]
    for uid in sorted(resolved_ids):
        for line in live.splitlines():
            if uid not in line:
                continue
            # The tally line legitimately says "37 unresolved, 2 resolved (U-20,
            # U-34)". It is the one place both words belong on one line.
            if "項 unresolved" in line and "項 resolved" in line:
                continue
            hit = next((p for p in _OPEN_PHRASES if p in line), None)
            check(f"{doc} does not call {uid} open",
                  hit is None,
                  f"{doc} says {uid} is `{hit}` but the registry marks it "
                  f"RESOLVED: {line.strip()[:150]}")

# Deviation ids must agree across the machine spec and every human table that
# lists them. This went stale the moment D-7 was added: three documents said
# "five deviations" while graph_spec held six, and a reader following the
# README would have omitted a real behavioural deviation from the report.
# The Mermaid flow in 01_GRAPH_SPEC is the only picture anyone actually looks
# at, and nothing kept it honest. It sat three rounds behind: no n05b, no
# frozen/trainable branch, no n10b, and n09b hanging off G3 alone. A diagram
# that omits a node is worse than no diagram, because it answers the question
# confidently.
spec_md = (DOCS / "01_GRAPH_SPEC.md").read_text()
mermaid = "\n".join(
    blk for blk in re.findall(r"```mermaid(.*?)```", spec_md, re.S)
)
for nid in nodes:
    short = nid.split("_")[0]
    check(f"mermaid shows {nid}", re.search(rf"\b{re.escape(short)}\b", mermaid) is not None,
          "node is in the registry but absent from the flow diagram")

# The node summary table in 01_GRAPH_SPEC is the human-readable index of the
# graph, and it sat three rounds behind the registry -- no n05b, no n10b. The
# Mermaid check below covers the picture; this covers the table.
spec_md_body = (DOCS / "01_GRAPH_SPEC.md").read_text()
for nid in nodes:
    if nid.startswith("G"):
        continue
    check(f"node table lists {nid}", f"`{nid}`" in spec_md_body,
          "node is in the registry but absent from 01_GRAPH_SPEC's prose")

dev_ids = {d["id"] for d in spec["boundary"]["deviations"]}
cond_ids = {d["id"] for d in spec["boundary"].get("conditional_deviations", [])}
for name in ("README.md", "02_BUILD_STEPS.md"):
    body = (DOCS / name).read_text()
    listed = set(re.findall(r"\|\s*\*\*(D-[0-9]+)\*\*", body))
    check(f"{name} deviation ids", listed == dev_ids | cond_ids,
          f"lists {sorted(listed)}, graph_spec has {sorted(dev_ids | cond_ids)}")
root = (DOCS.parents[1] / "README.md").read_text()
# BUG FIX 2026-08-22: the pattern was `D-[0-9]`, single digit. It could not
# match `D-10` and above at all, so the mirror tables would silently appear to
# be missing every two-digit id. Latent until D-9..D-12 were registered.
listed = set(re.findall(r"\|\s*\*\*(D-[0-9]+)\*\*", root))
check("root README deviation ids", listed == dev_ids | cond_ids,
      f"lists {sorted(listed)}, graph_spec has {sorted(dev_ids | cond_ids)}")

# An implemented node must reference every channel it declares writing. Both
# implemented nodes failed this: n02 and n03 each listed run_progress and
# cost_ledger in `writes` and emitted neither, which is the same
# declared-but-not-executed defect the spec reviews kept finding one level up.
# Only nodes carrying an IMPLEMENTS-NODE marker are checked -- a node with no
# code cannot be accused of not writing anything.
impl_src = {}
for d in ("metafind", "tools", "setup"):
    for f in (DOCS.parents[1] / d).rglob("*"):
        if f.suffix not in (".py", ".sh"):
            continue
        body = f.read_text(errors="replace")
        for nid in re.findall(r"^# IMPLEMENTS-NODE: (\S+)$", body, re.M):
            impl_src[nid] = body
for nid, body in impl_src.items():
    if nid not in nodes:
        continue
    for ch in nodes[nid].get("writes", []) or []:
        check(
            f"{nid} emits {ch}",
            ch in body,
            f"the registry says this node writes `{ch}`, but its source never "
            f"mentions it",
        )

test_count = sum(
    len(re.findall(r"^def test_", p.read_text(), re.M))
    for p in (DOCS.parents[1] / "tests").glob("test_*.py")
)
m2 = re.search(r"(\d+) 個測試函式涵蓋", readme_txt)
check("README unit-test count", m2 is not None and int(m2.group(1)) == test_count,
      f"README says {m2.group(1) if m2 else '?'}, tests/ defines {test_count}")
# --- docs/audit must not drift from the inventory it is built on ------------
# Added after the C1 decision left five separate documents still calling U-26
# unresolved. Reading them by eye found three; a script found the other two.
_audit = DOCS.parent / "audit"
if (_audit / "formula_inventory_validation.json").exists():
    _val = json.loads((_audit / "formula_inventory_validation.json").read_text())
    _n = _val["formulas_total"]
    _per_paper = {s["display_formulas"] for s in _val["summary"].values()}
    for _md in sorted(_audit.glob("*.md")):
        _txt = _md.read_text()
        for _m in re.finditer(r"\*\*(\d+) (?:display )?formulas?\*\*|(\d+) 條公式", _txt):
            _got = int(_m.group(1) or _m.group(2))
            check(f"{_md.name} formula count",
                  _got == _n or _got in _per_paper,
                  f"claims {_got} formulas; the inventory holds {_n} "
                  f"(per-paper {sorted(_per_paper)})")
    # A resolved UNKNOWN may not still be described as open in the audit either.
    for _md in sorted(_audit.glob("*.md")):
        _txt = _md.read_text()
        for _rid in sorted(resolved_ids):
            for _line in _txt.split("\n"):
                if _rid in _line and re.search(r"\bopen\b|unresolved", _line, re.I) \
                        and "decided" not in _line.lower():
                    check(f"{_md.name} calls {_rid} open",
                          False,
                          f"{_rid} is RESOLVED in the registry: {_line.strip()[:90]}")

for nid in implemented:
    check(f"IMPLEMENTS-NODE {nid}", nid in {n["id"] for n in registry["nodes"]},
          "marker names a node that is not in the registry")


# --- 8a. `channel.field` in a gate criterion must be a field of that channel
# G3 kept demanding stage1_protocol.image_aggregation / .text_serialization /
# .clip_train_scope for a whole round after those three fields moved to
# stage1_encoding_protocol. The gate read the right channels and named the
# right channels, so check 8 passed -- but it was validating a schema that no
# longer existed, and it would have rejected a correctly resolved pair. A gate
# is the barrier in front of the spend; it has to know the current shape of
# what it guards.
FIELD_RE = re.compile(r"\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\b")
# `type:` strings look like "{status: a|b, fusion, tower_sharing: x|y, ...}".
# Take the identifier before an optional `:` in each comma-separated part.
channel_fields: dict[str, set[str]] = {}
for name, ch in channels.items():
    t = str(ch.get("type", ""))
    # Take the {...} wherever it sits. Requiring the type to START with "{"
    # skipped every `map[uid -> {...}]` channel, which is most of them -- so
    # this check and 8a3 silently covered nothing for pointclouds, renders,
    # text_image_embeddings, post_stage1_embeddings and the rest.
    m = re.search(r"\{(.*)\}", t, re.S)
    if not m:
        continue
    channel_fields[name] = {
        part.split(":")[0].strip()
        for part in m.group(1).split(",")
        if part.split(":")[0].strip().isidentifier()
    }

for g in plan["level_3_gates"]:
    gid = g["gate_id"]
    for blob in (str(g.get("criterion", "")), str(g.get("scope_note", ""))):
        for chan, fld in FIELD_RE.findall(blob):
            if chan not in channel_fields:
                continue
            check(
                f"gate {gid} field {chan}.{fld}",
                fld in channel_fields[chan],
                f"channel `{chan}` has no field `{fld}`; it declares "
                f"{sorted(channel_fields[chan])}",
            )

for dp in spec["routing"]:
    for inp in dp.get("inputs", []):
        chan, _, fld = str(inp).partition(".")
        # `sem_edge_cache.contains(hash)` is a PREDICATE over the channel, not
        # a field of it. Anything with a call is a computation and is checked
        # only for naming a real channel.
        if "(" in str(inp):
            check(
                f"routing {dp['decision_point']} predicate {inp}",
                chan in channels,
                f"predicate is over `{chan}`, which is not a channel",
            )
        elif fld and chan in channel_fields:
            check(
                f"routing {dp['decision_point']} input {inp}",
                fld in channel_fields[chan],
                f"channel `{chan}` has no field `{fld}`",
            )


# --- 8a3. every routing input must be traceable to something -------------
# G2 routes on centroid_offset, max_radius and per_axis_variance while the
# pointclouds channel type declared only {path, sha256, n_points} -- and the
# producer wrote neither path nor sha256. A gate was configured to decide on
# quantities that no channel declared and no node produced, and check 8a could
# not see it because routing names them as bare identifiers rather than
# `channel.field`.
# Three legitimate origins, and a routing input must have one of them:
#   * the name of a channel the gate reads
#   * a field of a channel the gate reads
#   * a quantity the gate COMPUTES, declared in derived_inputs
# The third exists so that "the gate works it out" is written down rather than
# assumed; without it this check would either be wrong or unenforceable.
for dp in spec["routing"]:
    gid = dp["decision_point"]
    if gid not in nodes:
        continue
    reads = set(nodes[gid].get("reads", []) or [])
    available = set(reads)
    for ch in reads:
        available |= channel_fields.get(ch, set())
    derived_specs = dp.get("derived_inputs", []) or []
    declared_derived = {d["name"] for d in derived_specs}
    # A derived quantity must name the channel it is COMPUTED FROM, and the
    # gate must read that channel. Without this the category was a way to
    # declare anything: G5 routed on `verdict_completeness` while reading only
    # gate_records, audit_records, degraded_flags and cost_ledger, none of
    # which holds a per-cell verdict. "The gate works it out" was
    # unfalsifiable.
    for d in derived_specs:
        check(
            f"derived source {gid}/{d['name']}",
            d.get("from") in reads,
            f"is declared derived from `{d.get('from')}`, which {gid} does not read",
        )
    for inp in dp.get("inputs", []):
        base = str(inp).split(".")[0]
        check(
            f"routing input {gid}/{inp}",
            base in available or str(inp) in declared_derived,
            f"is neither a channel {gid} reads, nor a field of one, nor listed "
            f"in derived_inputs",
        )
    for d in declared_derived:
        check(
            f"derived_inputs {gid}/{d}",
            d not in available,
            "is declared as computed by the gate but is also a channel or a "
            "field of one; one of the two is wrong",
        )


# --- 8a2. an `any` join group must have >1 edge and all of them guarded ---
# n09_build_splits joined `all` over the frozen-cache edge and would have
# joined the same way over a trainable-route edge, which is how the trainable
# reading of U-34 became structurally unreachable: the join demanded a cache
# that reading forbids producing. The repair is a branch, and a branch is only
# real if every arm carries a guard -- one unguarded arm always fires and the
# other arms are decoration.
for j in spec["join_policies"]:
    node = j["node"]
    for grp in j.get("groups", []):
        if grp.get("policy") != "any":
            continue
        arms = [e for e in edges if e["to"] == node and e.get("join_group") == grp["name"]]
        if not arms:
            # Subgraph-internal nodes (sg1_generate, sg4_encode_layout, ...)
            # are fed by loop edges that the top-level `edges:` list does not
            # contain. Nothing to compare against here.
            continue
        check(
            f"any-group {node}/{grp['name']} size",
            len(arms) > 1,
            f"policy `any` over {len(arms)} edge(s) is just `all` with a weaker name",
        )
        check(
            f"any-group {node}/{grp['name']} guarded",
            all(e.get("guard") for e in arms),
            "unguarded arm: " + ", ".join(e["id"] for e in arms if not e.get("guard")),
        )


# --- 8b. the dependency DAG must agree with the edge list ----------------
# n09_build_splits kept `depends_on: n08_semantic_edges` for a whole round after
# the Objaverse/ProcTHOR branches were split. The edges and the node's `reads`
# were both corrected; this third declaration was not, so the machine-readable
# dependency still said Stage 1 waits on Qwen semantic edges.
edge_preds = defaultdict(set)
for e in edges:
    if e["from"] in nodes and e["to"] in nodes and e.get("kind") not in ("feedback", "error", "escalation"):
        edge_preds[e["to"]].add(e["from"])

def _deps(dep) -> tuple[set[str], set[str]]:
    """(required, conditional) predecessors of one DAG entry."""
    req = set(dep.get("depends_on", []) or [])
    cond = {c["node"] for c in dep.get("conditional_depends_on", []) or []}
    return req, cond


for dep in spec["dependencies"]["dag"]:
    n = dep["node"]
    required, conditional = _deps(dep)
    declared = required | conditional
    check(
        f"dag matches edges for {n}",
        declared == edge_preds[n],
        f"depends_on {sorted(declared)} but incoming edges come from {sorted(edge_preds[n])}",
    )


# --- 8b2. what a gate reads must actually arrive on an incoming edge ------
# essgnn_edge_protocol was added to G6's `reads` and to its criterion while
# e13c still carried only stage2_protocol and stage2_pairing. Channel
# declarations said one thing and the edge payload another, and nothing
# compared them.
incoming_payload = defaultdict(set)
for e in edges:
    if e["to"] in nodes:
        incoming_payload[e["to"]].update(e.get("carries", []))

BROADCAST = {"run_progress", "cost_ledger", "degraded_flags", "gate_records",
             "audit_records", "quarantine"}
for gid in {g["gate_id"] for g in plan["level_3_gates"]}:
    if gid not in nodes:
        continue
    for ch in set(nodes[gid].get("reads", []) or []) - BROADCAST:
        check(
            f"gate {gid} receives {ch}",
            ch in incoming_payload[gid],
            "is read by the gate but no incoming edge carries it",
        )


# --- 8b3. NODE-READ-ANCESTOR: every node's reads must be produced upstream -
# The check above runs only over gates, and that gap hid a whole class of
# defect: an ordinary node could declare `reads: [X]` while nothing in its
# dependency closure writes X. n13_train_stage2 read post_stage1_embeddings
# with n10b absent from its depends_on; n09b read scene_graphs with only G3
# upstream, and G3 was deliberately decoupled from the ProcTHOR branch so it
# cannot imply n07 has run; n15b read stage2_protocol and scene_graphs with
# neither writer among its ancestors. Each worked only because the execution
# layering happened to put the writer earlier -- an accident of scheduling,
# not a contract.
dag_pred = {d["node"]: (set(d.get("depends_on", []) or [])
                        | {c["node"] for c in d.get("conditional_depends_on", []) or []})
            for d in spec["dependencies"]["dag"]}


def ancestors(n, _seen=None):
    _seen = _seen if _seen is not None else set()
    for p_ in dag_pred.get(n, ()):
        if p_ not in _seen:
            _seen.add(p_)
            ancestors(p_, _seen)
    return _seen


# Channels every node may read without a producer edge: append-only telemetry
# and the global immutable inputs that exist before the graph starts.
GLOBAL_INPUTS = {"asset_manifest", "variant_registry"}
for nid, n in nodes.items():
    if nid not in dag_pred:
        continue  # sources and subgraph-internal nodes
    anc = ancestors(nid)
    for ch in set(n.get("reads", []) or []) - BROADCAST - GLOBAL_INPUTS:
        writers = set(channels.get(ch, {}).get("writers", []) or [])
        if not writers:
            continue
        check(
            f"read-ancestor {nid}/{ch}",
            bool(writers & anc) or nid in writers,
            f"reads `{ch}` but none of its writers {sorted(writers)} is a "
            f"dependency ancestor; the value is available only by scheduling luck",
        )


# --- 8b4. a conditional dependency must actually be skippable -------------
# Declaring n06 conditional is only meaningful if n09 can run without it. If
# every arm carrying n06's payload sits in an `all` join group, the guard stops
# n06 and the join then waits for it regardless -- the same deadlock, now with
# a schema field asserting otherwise.
for dep in spec["dependencies"]["dag"]:
    node = dep["node"]
    policies = {j["node"]: j for j in spec["join_policies"]}
    for cond in dep.get("conditional_depends_on", []) or []:
        pred = cond["node"]
        arms = [e for e in edges if e["to"] == node and e["from"] == pred]
        check(
            f"conditional dep {node}<-{pred} guarded",
            all(e.get("guard") for e in arms) and bool(arms),
            "a conditional dependency needs at least one incoming edge and "
            "every one of them guarded",
        )
        groups = {e.get("join_group", "default") for e in arms}
        declared = {g["name"]: g for g in policies.get(node, {}).get("groups", [])}
        check(
            f"conditional dep {node}<-{pred} skippable",
            all(declared.get(g, {}).get("policy") == "any" for g in groups),
            f"its payload arrives in join group(s) {sorted(groups)} whose policy "
            f"is not `any`, so {node} waits for a node that will not run",
        )


# --- 8c. gate evidence must name checks that exist -----------------------
# G3 cited L2-LEAK for a whole round after that check was split into
# L2-LEAK-OBJECT and L2-LEAK-SCENE. A gate resting on a check that does not
# exist is a gate resting on nothing.
check_ids = {c["id"] for c in plan["level_1"]} | {c["id"] for c in plan["level_2"]}
for g in plan["level_3_gates"]:
    for ev in g.get("evidence", []):
        check(
            f"{g['gate_id']} evidence {ev}",
            ev in check_ids,
            "names a check that does not exist in level_1 or level_2",
        )

for c in plan["level_2"]:
    gate = c.get("cited_by_gate")
    if gate:
        check(
            f"{c['id']} cited_by_gate",
            gate in {g["gate_id"] for g in plan["level_3_gates"]},
            f"cites {gate}, which is not a gate",
        )


# --- 9. execution order respects declared dependencies -------------------
# n20 and n21 were once scheduled in the same layer while n21 depended on n20.
layer_of: dict[str, int] = {}
for order, layer in enumerate(spec["execution_order"]["layers"]):
    for n in layer.get("parallel_nodes", []):
        layer_of[n] = order
    for n in layer.get("gates", []):
        layer_of[n] = order + 0.5  # a gate is evaluated after its layer's nodes

for dep in spec["dependencies"]["dag"]:
    n, ds = dep["node"], dep["depends_on"]
    if n not in layer_of:
        FAILURES.append(f"{n} has dependencies but no execution layer")
        CHECKS += 1
        continue
    for d in ds:
        CHECKS += 1
        if d in layer_of and not layer_of[d] < layer_of[n]:
            FAILURES.append(
                f"execution order: {n} (layer {layer_of[n]}) must come after "
                f"{d} (layer {layer_of[d]})"
            )

for n in nodes:
    check(f"scheduled {n}", n in layer_of, "node has no execution layer")


# --- 10. join policies name real nodes and real groups -------------------
edge_groups = defaultdict(set)
for e in edges:
    edge_groups[e["to"]].add(e.get("join_group", "default"))

for j in spec["join_policies"]:
    node = j["node"]
    check(f"join node {j.get('node')}", node in all_nodes, "unknown node")
    if node in all_nodes and node in edge_groups:
        for grp in j.get("groups", []):
            g = grp["name"]
            check(
                f"join group {node}/{g}",
                g in edge_groups[node],
                f"no incoming edge carries join_group {g}",
            )
        check(
            f"join coverage {node}",
            edge_groups[node] <= {grp["name"] for grp in j.get("groups", [])},
            f"incoming groups {sorted(edge_groups[node])} exceed the declared policy",
        )


# --- 10b. no duplicate mapping keys anywhere in the YAML files -----------
# L2-PC-ULIP-REF carried two `note:` keys. PyYAML keeps the last silently, so a
# whole paragraph of justification was discarded at parse time -- invisible to
# every check that reads the parsed document instead of the file.
class _DupDetect(yaml.SafeLoader):
    pass


def _no_dupes(loader, node, deep=False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            FAILURES.append(
                f"duplicate key `{key}` at line {key_node.start_mark.line + 1} "
                f"of {getattr(loader, '_fname', '?')} -- YAML silently keeps the last"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_DupDetect.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes
)

for fname in ("graph_spec.yaml", "node_registry.yaml", "validation_plan.yaml"):
    loader = _DupDetect((DOCS / fname).read_text())
    loader._fname = fname
    try:
        loader.get_single_data()
    finally:
        loader.dispose()
    CHECKS += 1


# --- 11. UNKNOWN identifiers resolve to the registry ---------------------
# 00_FINDINGS once carried its own U-01..U-05 with different meanings.
spec_md = text("01_GRAPH_SPEC.md")
registry_ids = set(re.findall(r"^\| \*\*`?(U-[0-9a-z]+)`?\*\* \|", spec_md, re.M))
registry_ids |= set(re.findall(r"^\| \*\*(U-[0-9a-z]+)\*\* \|", spec_md, re.M))
check("UNKNOWN registry non-empty", bool(registry_ids), "found no U-ids in 01_GRAPH_SPEC")

for f in sorted(DOCS.glob("*.md")) + sorted(DOCS.glob("*.yaml")):
    if f.name == "01_GRAPH_SPEC.md":
        continue
    body = f.read_text()
    if "SUPERSEDED" in body:
        body = body.split("SUPERSEDED")[0]
    for uid in set(re.findall(r"\bU-[0-9]{2}[ab]?\b", body)):
        check(
            f"{f.name} {uid}",
            uid in registry_ids,
            f"{uid} is not in the 01_GRAPH_SPEC UNKNOWN registry",
        )


# --- 11b. code must use the same UNKNOWN numbering as the registry --------
# metafind/models/*.py once carried its own U-07..U-12, where U-08 meant "all
# modalities masked" while the registry's U-08 meant "how Stage 2 samples are
# built". Same identifier, different fact, in two places an agent will grep.
CODE = DOCS.parents[1]
for f in sorted(CODE.glob("metafind/**/*.py")) + sorted(CODE.glob("tests/*.py")):
    if "vendor" in f.parts or "third_party" in f.parts:
        continue
    for uid in set(re.findall(r"\bU-[0-9]{2}[ab]?\b", f.read_text())):
        check(
            f"{f.relative_to(CODE)} {uid}",
            uid in registry_ids,
            f"{uid} is not in the 01_GRAPH_SPEC UNKNOWN registry",
        )


# --- 12. counts agree across documents ------------------------------------
n_l1 = len(plan["level_1"])
n_l2 = len(plan["level_2"])
n_gates = len(plan["level_3_gates"])
n_channels = len(channels)
n_nodes = len(nodes)
n_edges = len(edges)

check(
    "validation_plan.gate_discipline.total_l1",
    plan["gate_discipline"].get("total_l1") == n_l1,
    f"declares {plan['coverage_check'].get('total_l1')}, actual {n_l1}",
)
check(
    "validation_plan.gate_discipline.total_l2",
    plan["gate_discipline"].get("total_l2") == n_l2,
    f"declares {plan['coverage_check'].get('total_l2')}, actual {n_l2}",
)

check(
    "validation_plan.gate_discipline.total_gates",
    plan["gate_discipline"].get("total_gates") == n_gates,
    f"declares {plan['gate_discipline'].get('total_gates')}, actual {n_gates}",
)

counts = {
    "L1": n_l1,
    "L2": n_l2,
    "gates": n_gates,
    "channels": n_channels,
    "nodes": n_nodes,
    "edges": n_edges,
}
for fname in ("README.md", "01_GRAPH_SPEC.md", "02_BUILD_STEPS.md"):
    # The correction logs quote superseded numbers on purpose.
    body = re.split(r"^#+ .*修正紀錄", text(fname), flags=re.M)[0]
    for label, pat in (
        ("L1", r"(\d+)\s*個\s*L1"),
        ("L2", r"(\d+)\s*個\s*L2"),
        ("gates", r"(\d+)\s*個\s*gate"),
        ("channels", r"(\d+)\s*個\s*state channel"),
        ("nodes", r"(\d+)\s*個節點"),
        ("edges", r"(\d+)\s*條邊"),
    ):
        for m in re.finditer(pat, body):
            check(
                f"{fname} {label} count",
                int(m.group(1)) == counts[label],
                f"says {m.group(1)}, actual {counts[label]}",
            )

# the gate chain drawn in prose must list exactly the gates that exist
gate_ids = {g["gate_id"] for g in plan["level_3_gates"]}
for fname in ("README.md", "01_GRAPH_SPEC.md"):
    named = set(re.findall(r"\bG\d+(?=[_\s一-鿿])",
                           re.split(r"^#+ .*修正紀錄", text(fname), flags=re.M)[0]))
    short = {g.split("_")[0] for g in gate_ids}
    check(
        f"{fname} gate mentions",
        named <= short,
        f"mentions {sorted(named - short)} which are not gates",
    )
    check(
        f"{fname} gate coverage",
        short <= named,
        f"never mentions {sorted(short - named)}",
    )


# --- 13. every mutate node sits behind a gate ----------------------------
gate_layers = sorted(layer_of[g] for g in gate_ids if g in layer_of)
for nid, n in nodes.items():
    if n.get("role") == "mutate":
        check(
            f"mutate {nid} gated",
            any(gl < layer_of.get(nid, -1) for gl in gate_layers),
            "mutate node has no gate before it",
        )


# --- 14. required audits never block -------------------------------------
for ra in plan["required_audits"]:
    blob = str(ra)
    check(
        f"{ra['audit_id']} non-blocking",
        ra.get("blocks") is False or "never block" in blob or "does not block" in blob,
        "a Required Audit must be explicitly non-blocking",
    )


# --------------------------------------------------------------------------
print(f"channels {n_channels}  nodes {n_nodes}  edges {n_edges}  "
      f"gates {n_gates}  L1 {n_l1}  L2 {n_l2}")
print(f"{CHECKS} checks")
if FAILURES:
    print(f"\n{len(FAILURES)} FAILURES\n")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("all pass")
