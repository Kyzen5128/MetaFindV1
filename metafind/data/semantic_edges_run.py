"""Generate and encode the semantic edges of every ProcTHOR scene graph.

# IMPLEMENTS-NODE: n08_semantic_edges

Writes ``sem_edge_cache`` (sentences, embeddings and the degraded flags), and
``quarantine`` / ``run_progress`` / ``cost_ledger`` via runlog.

SG2, in the order the spec gives it:

    sg2_select_pairs -> sg2_cache_lookup -> sg2_generate -> sg2_validate
                                                |               |
                                                +-- C2 repair (2) --+
                                                                |
                                       degraded (flagged)  <----+
                                                |
                                          sg2_encode -> sg2_reduce

Pair selection is n07's: it wrote ``sem_edge_ids`` per house, and this node
reads them. That is where [U-06] was decided, not here.

Why this runs in three phases rather than one loop
--------------------------------------------------

MEASURED: 12,000 houses hold 4,128,637 semantic edges but only 4,242 distinct
description pairs -- a 99.90% cache hit rate. Generating per edge would make
4.1 million LLM calls to produce four thousand distinct answers. So phase 1
enumerates the distinct pairs, phase 2 generates a sentence for each, and phase
3 encodes them in one batch. Per-house attachment is then a dictionary lookup
with no model in the loop at all.

The sentences file is appended and fsynced as each one lands, so a run killed
at 3,000 pairs resumes at 3,000 rather than at zero.

Why an exhausted edge is flagged and not zero-filled
----------------------------------------------------

[L1-SEMEDGE-NO-ZEROFILL] A zero vector is a perfectly valid point in the
embedding space. Downstream, an edge carrying zeros is indistinguishable from
one carrying a real relation that happened to encode near the origin -- the
failure would travel silently into every Table 2 and Table 3 number. So an
exhausted edge gets ``degraded: true`` and no embedding, and [U-30] the tensor
contract for filling f_h's e slots is n13's problem to state explicitly rather
than this node's to paper over.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from metafind import paths, runlog

# BEFORE transformers or torch reaches this process. HF_HOME is read at IMPORT
# time; setting it later sends a 16 GB download to the 100 GB root partition
# while a complete copy sits on the data volume. n05 learned this the slow way.
paths.setup_env()

from metafind.data.semantic_edges import (  # noqa: E402
    MAX_ATTEMPTS,
    PROMPT_VERSION,
    SemanticEdgeError,
    build_relation_prompt,
    build_repair_prompt,
    cache_key,
    parse_sentence,
    validate_sentence,
)

NODE = "n08_semantic_edges"

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # D-2's stand-in, text-only here

# [U-06, the "width of e_ij" half] The paper says "a frozen text encoder (e.g.,
# CLIP or BERT)" and stops. CLIP ViT-B/32 gives 512 dimensions; the bigG variant
# also cached here gives 1280, which would make e_ij wider than the node feature
# it is concatenated with in f_h : R^(2d+1+e) -> R^d. Recorded as our choice.
TEXT_ENCODER = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
TEXT_ENCODER_VERSION = "clip-vit-b32-laion2b-s34b-b79k"

MAX_NEW_TOKENS = 64
ENCODE_BATCH = 256

SENTENCES_PATH = paths.OUTPUTS / "sem_edge_sentences.jsonl"
EMBEDDINGS_PATH = paths.OUTPUTS / "sem_edge_embeddings.npz"
CACHE_PATH = paths.OUTPUTS / "sem_edge_cache.json"


# --- phase 1: which distinct pairs exist ----------------------------------

def collect_pairs(text_map: dict, limit: int | None = None) -> dict[str, tuple[str, str]]:
    """Every distinct description pair across the scene graphs, keyed by cache key.

    Reads the sidecars rather than the index because the index carries counts,
    not the pairs themselves.
    """
    pairs: dict[str, tuple[str, str]] = {}
    files = sorted(glob.glob(str(paths.SCENE_GRAPHS / "*.json")))
    if limit:
        files = files[:limit]
    for path in files:
        try:
            graph = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nodes = graph["nodes"]
        for i, j in graph["sem_edge_ids"]:
            ti = text_map.get(str(nodes[i]["asset_id"]))
            tj = text_map.get(str(nodes[j]["asset_id"]))
            if ti is None or tj is None:
                continue
            a, b = sorted((ti["text"], tj["text"]))
            pairs.setdefault(
                cache_key(a, b, PROMPT_VERSION, LLM_MODEL, TEXT_ENCODER_VERSION),
                (a, b),
            )
    return pairs


# --- phase 2: a sentence per pair -----------------------------------------

class RelationWriter:
    """Qwen, text-only. One instance per process."""

    def __init__(self, model_id: str = LLM_MODEL, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                # Greedy: a C2 retry must differ because the PROMPT carries the
                # error back, not because the sampler rolled differently.
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(trimmed[0], skip_special_tokens=True)


def write_one(gen: RelationWriter, desc_a: str, desc_b: str) -> tuple[str | None, str | None]:
    """C2 for one pair. Returns ``(sentence, error)`` -- exactly one is None."""
    prompt = build_relation_prompt(desc_a, desc_b)
    current, last_error, last_raw = prompt, "", ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = gen.generate(current)
        last_raw = raw
        try:
            return validate_sentence(parse_sentence(raw)), None
        except SemanticEdgeError as exc:
            last_error = str(exc)
            if attempt < MAX_ATTEMPTS:
                current = build_repair_prompt(prompt, last_error, raw)

    return None, f"{last_error} | last response: {last_raw[:200]}"


def load_sentences() -> dict[str, dict]:
    """Resume point: whatever earlier runs already settled."""
    if not SENTENCES_PATH.exists():
        return {}
    done: dict[str, dict] = {}
    for line in SENTENCES_PATH.read_text().splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed run
            done[rec["key"]] = rec
    return done


def append_sentence(rec: dict) -> None:
    with SENTENCES_PATH.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --- phase 3: encode ------------------------------------------------------

def encode_sentences(sentences: list[str], model_id: str = TEXT_ENCODER) -> np.ndarray:
    """[PAPER 2.5] The frozen text encoder that produces e_ij.

    Frozen means no gradients and no training here; the embeddings are computed
    once and cached. L2-normalised, which is CLIP's own convention for its text
    tower and keeps e_ij on the same scale across pairs.
    """
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = CLIPTextModelWithProjection.from_pretrained(model_id).eval().cuda()
    for p in enc.parameters():
        p.requires_grad_(False)

    out = []
    with torch.no_grad():
        for start in range(0, len(sentences), ENCODE_BATCH):
            batch = sentences[start : start + ENCODE_BATCH]
            inputs = tok(batch, padding=True, truncation=True, max_length=77,
                         return_tensors="pt").to("cuda")
            emb = enc(**inputs).text_embeds
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


# --- phase 4: assemble the cache ------------------------------------------

def build_cache(settled: dict[str, dict], embeddings_uri: str) -> dict[str, dict]:
    """[L1-SEMEDGE-NO-ZEROFILL] Settled sentences into sem_edge_cache entries.

    A degraded pair gets an entry that SAYS it is missing. Two alternatives were
    both rejected: a zero vector, which is indistinguishable downstream from a
    real relation that encoded near the origin; and an absent key, which the
    next run would read as "not computed yet" and retry forever against a bound
    that has already been exhausted.

    Lives here rather than inline in ``main`` so the test exercises the code
    that actually runs, not a second copy of it.
    """
    good = [(k, r) for k, r in settled.items() if not r["degraded"]]
    cache = {
        key: {"sentence": rec["sentence"],
              "embedding_uri": f"{embeddings_uri}#{idx}",
              "degraded": False}
        for idx, (key, rec) in enumerate(good)
    }
    for key, rec in settled.items():
        if rec["degraded"]:
            cache[key] = {"sentence": None, "embedding_uri": None,
                          "degraded": True, "semantic_edge_missing": True}
    return cache


# --- driver ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-houses", type=int, help="scene graphs to scan (smoke)")
    ap.add_argument("--limit-pairs", type=int, help="pairs to generate (smoke)")
    ap.add_argument("--skip-encode", action="store_true")
    args = ap.parse_args()

    text_path = paths.OUTPUTS / "procthor_object_text.json"
    if not text_path.exists():
        print(f"{text_path} not found -- run n07_scene_graphs first", flush=True)
        return 2

    text_map = json.loads(text_path.read_text())
    pairs = collect_pairs(text_map, args.limit_houses)
    print(f"{len(pairs):,} distinct description pairs", flush=True)

    settled = load_sentences()
    todo = [k for k in pairs if k not in settled]
    if args.limit_pairs:
        todo = todo[: args.limit_pairs]
    print(f"{len(settled):,} already settled, {len(todo):,} to generate", flush=True)

    degraded, started = 0, time.time()
    if todo:
        gen = RelationWriter()
        with runlog.run_progress(NODE):
            for n, key in enumerate(todo, 1):
                a, b = pairs[key]
                sentence, error = write_one(gen, a, b)
                if sentence is None:
                    # [L1-SEMEDGE-NO-ZEROFILL] flagged, never zero-filled
                    append_sentence({"key": key, "desc_a": a, "desc_b": b,
                                     "sentence": None, "degraded": True,
                                     "reason": error})
                    runlog.quarantine(NODE, [{
                        "pair": [a, b], "failure_class": "MODEL_RECOVERABLE",
                        "terminated_by": "repair_budget", "attempts": MAX_ATTEMPTS,
                        "schema_errors": (error or "")[:400],
                    }])
                    degraded += 1
                else:
                    append_sentence({"key": key, "desc_a": a, "desc_b": b,
                                     "sentence": sentence, "degraded": False})
                if n % 200 == 0:
                    rate = n / max(time.time() - started, 1e-9) * 60
                    print(f"  [{n:5d}/{len(todo)}] {rate:.0f}/min, "
                          f"degraded {degraded}", flush=True)
        del gen  # the encoder needs the memory back

    settled = load_sentences()
    good = [(k, r) for k, r in settled.items() if not r["degraded"]]
    if args.skip_encode:
        print(f"{len(good):,} sentences, encoding skipped")
        return 0

    embeddings = encode_sentences([r["sentence"] for _, r in good])
    np.savez_compressed(
        EMBEDDINGS_PATH,
        keys=np.array([k for k, _ in good]),
        embeddings=embeddings.astype(np.float32),
    )

    cache = build_cache(settled, str(EMBEDDINGS_PATH))

    tmp = CACHE_PATH.with_suffix(".json.part")
    with tmp.open("w") as fh:
        json.dump({"llm_model": LLM_MODEL, "text_encoder": TEXT_ENCODER,
                   "text_encoder_version": TEXT_ENCODER_VERSION,
                   "prompt_version": PROMPT_VERSION, "edge_dim": int(embeddings.shape[1]),
                   "entries": cache}, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(CACHE_PATH)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       pairs_generated=len(todo), llm_calls=len(todo),
                       degraded_edges=degraded)
    print(f"\n{len(cache):,} cache entries ({degraded:,} degraded), "
          f"e_ij dim {embeddings.shape[1]} -> {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
