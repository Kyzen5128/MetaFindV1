#!/usr/bin/env python3
"""Structural checker for docs/graph/.

Every check here exists because it caught a real bug. Running it is the only way
the six specification documents stay consistent with each other; reading them
does not scale.

    python3 tools/check_graph.py

Exit 0 = all checks pass, 1 = at least one failure.
"""

from __future__ import annotations

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
PSEUDO = {"RA-1", "RA-2", "RA-3"}  # audits write records but are not graph nodes
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


# --- 8b. the dependency DAG must agree with the edge list ----------------
# n09_build_splits kept `depends_on: n08_semantic_edges` for a whole round after
# the Objaverse/ProcTHOR branches were split. The edges and the node's `reads`
# were both corrected; this third declaration was not, so the machine-readable
# dependency still said Stage 1 waits on Qwen semantic edges.
edge_preds = defaultdict(set)
for e in edges:
    if e["from"] in nodes and e["to"] in nodes and e.get("kind") not in ("feedback", "error", "escalation"):
        edge_preds[e["to"]].add(e["from"])

for dep in spec["dependencies"]["dag"]:
    n, declared = dep["node"], set(dep["depends_on"])
    check(
        f"dag matches edges for {n}",
        declared == edge_preds[n],
        f"depends_on {sorted(declared)} but incoming edges come from {sorted(edge_preds[n])}",
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
