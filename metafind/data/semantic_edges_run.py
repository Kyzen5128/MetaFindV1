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

# [USER 2026-08-28] 「我說過所有LLM模型都改成用gemma」. D-2 was re-pointed at gemma
# on 2026-08-24 (METAFIND_NOTEBOOK.md:147, 「D-2 改成 gemma」) and this line was
# not moved with it -- it still named the model D-2 pointed at BEFORE that
# ruling, so n08 would have written its sentences with a model the project had
# already stopped using, while citing the decision that stopped it.
#
# Imported rather than re-spelled: two nodes each holding their own literal is
# how the two drifted apart in the first place.
from metafind.data.annotate_run import MODEL_ID as LLM_MODEL_PATH  # noqa: E402

# The PATH loads the weights; the NAME goes in the cache key and the provenance
# record. Splitting them is not tidiness -- `MODEL_ID` is a local absolute path,
# and an earlier version of this line let that path into
# `cache_key(..., llm_model, ...)`, whose sha256 then depended on where the
# checkpoint happens to sit. That directory has already moved once
# (/mnt/data1/kyzen/models -> /home/kyzen/metafind/metafind_out), and moving it again
# would have invalidated all 4,242 cached sentences for an unchanged model.
# It also made `llm_model` in the record a path rather than a model identity.
# Caught by ESSGNN ENGINEER [487717] 2026-08-28; a same-machine smoke cannot
# catch it, because the path is identical on both runs.
LLM_MODEL = Path(LLM_MODEL_PATH).name          # 'gemma-4-12B-it'


# [U-20] ✅ USER-APPROVED 2026-08-27 (METAFIND_NOTEBOOK.md:937): the node and
# edge text encoder is **OpenCLIP ViT-bigG-14, 1280-d**. Kyzen's reason is
# consistency: Stage 1's 1280 is forced by the ULIP-2 checkpoint
# (`ulip_backbone.py:90` EMBED_DIM = 1280, projection shaped (768, 1280)), and
# one project should not hold two different text understandings.
#
# WITHDRAWN, and left visible rather than deleted so it cannot be re-derived:
# this constant used to be `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at 512, on the
# argument that "the tower's job is to be the retrieval space, ESSGNN's job is to
# keep geometry legible, and F8 says those pull in opposite directions". That
# argument was overruled on 2026-08-27.
#
# SCOPE, and this correction is the point [MASTER 2026-08-28]: F8's "a wide e_ij
# drowns the single ||x_i - x_j||^2 scalar" is a statement about f_h, the TWO-MLP
# layer (`essgnn.py:325`). Our family is `appendix_shared_msg`
# (`essgnn_arch_protocol.json`), which uses phi_e (`essgnn.py:422`), and there
# the measured direction is the OPPOSITE -- widening strengthened the geometric
# term. `tests/test_essgnn.py:255`
# (`test_f8_does_not_generalise_to_the_appendix_layer`) already exists to stop
# exactly this generalisation, and its docstring says so; this comment was
# written without reading it. The magnitudes once quoted for F8 are also not
# reproducible from the current test -- its `for seed in range(6)` never passes
# the seed, so six calls are one sample (METAFIND_NOTEBOOK.md 9.8). What
# survives is a qualitative finding about two_mlp only, and it is not a reason
# for n08 to emit a different encoder's vectors.
#
# Reached through `ULIPBackbone` rather than through `transformers`
# `CLIPTextModelWithProjection`: OBSERVED DATA, the only bigG in the local cache
# is `open_clip_model.safetensors` (no config.json, no HF-format weights), so a
# `from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")` would download a
# SECOND ~10 GB copy and encode with a different implementation of the same
# model. "One text understanding" is the whole reason U-20 was decided this way,
# so the tower's own weights are what must be used -- the same object n06 uses.
#
# [U-20 continued] This pins t_i's encoder as well, and pinning it here is the
# point. t_i and e_ij both feed f_h; letting n08 pick one encoder and a later
# node pick another would put ESSGNN's node and edge features in two unrelated
# semantic spaces for no stated reason. Whatever encodes t_i must be THIS model.
TEXT_ENCODER = "open_clip:ViT-bigG-14"       # via ULIPBackbone, not from_pretrained
TEXT_ENCODER_VERSION = "open-clip-vit-bigg-14-ulip2-tower"
EDGE_DIM = 1280

MAX_NEW_TOKENS = 64
ENCODE_BATCH = 256

SENTENCES_PATH = paths.OUTPUTS / "sem_edge_sentences.jsonl"
EMBEDDINGS_PATH = paths.OUTPUTS / "sem_edge_embeddings.npz"
CACHE_PATH = paths.OUTPUTS / "sem_edge_cache.json"
NODE_EMB_PATH = paths.OUTPUTS / "procthor_node_embeddings.npz"
NODE_EMB_RECORD = paths.OUTPUTS / "procthor_node_embeddings.json"


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
    """gemma, text-only. One instance per process.

    Loaded the way n05 loads it, not the way a text-only checkpoint would be:
    `gemma-4-12B-it`'s config declares `Gemma4UnifiedForConditionalGeneration`,
    so `AutoModelForCausalLM` + `AutoTokenizer` is the wrong pair for it. The
    processor also owns the chat template; the tokenizer alone does not.
    Text-only here means the content list carries no image, NOT that the model
    is a text-only one.
    """

    def __init__(self, model_id: str = LLM_MODEL_PATH, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        # [PAPER FACT `2methdology.tex:47`] "semantic edges are generated by
        # prompting a large language model (LLM) with object descriptions to
        # produce natural language relation sentences" -- the input to this node
        # is DESCRIPTIONS. Images do not appear in that definition.
        #
        # The guard exists because the loader below is the image-text-to-text
        # one (gemma-4-12B-it declares Gemma4UnifiedForConditionalGeneration and
        # will not load any other way), which puts "also pass the eleven views"
        # one line away and silently turns n08 into a different method. Nothing
        # would raise; the sentences would just quietly stop being the paper's.
        #
        # [MASTER 2026-08-28] WHAT THIS DOES NOT CATCH, stated so the next reader
        # does not mistake it for complete protection: it only inspects the
        # content list this function builds. Anyone calling `self.model.generate`
        # directly is not bound by it. Widening it would mean changing
        # RelationWriter's interface, which still would not stop a determined
        # caller and would leave the impression that it did -- worse than no
        # guard. The guard is for a slip; n08's text-only nature is carried by
        # the SPEC.
        content = [{"type": "text", "text": prompt}]
        non_text = sorted({c.get("type") for c in content} - {"text"})
        if non_text:
            raise SemanticEdgeError(
                f"n08 is text-only and was given {non_text}. "
                "2methdology.tex:47 defines semantic edges as an LLM prompted "
                "WITH OBJECT DESCRIPTIONS; feeding views here is a different "
                "method and must be recorded as a DEVIATION before it runs."
            )
        messages = [{"role": "user", "content": content}]
        # Same guard as n05: a template with no such variable raises on an
        # unexpected keyword, and a thinking model narrates instead of answering.
        template_kwargs: dict = {}
        if "enable_thinking" in (getattr(self.processor, "chat_template", "") or ""):
            template_kwargs["enable_thinking"] = False
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                # Greedy: a C2 retry must differ because the PROMPT carries the
                # error back, not because the sampler rolled differently.
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(trimmed[0], skip_special_tokens=True)


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

def encode_sentences(sentences: list[str]) -> np.ndarray:
    """[PAPER 2.5] The frozen text encoder that produces e_ij.

    Frozen means no gradients and no training here; the embeddings are computed
    once and cached. L2-normalised, which is CLIP's own convention for its text
    tower and keeps e_ij on the same scale across pairs.
    """
    import torch

    # [U-20] The tower's own text half, not a second copy of the same model
    # loaded through `transformers`. `train_scope="fuser_only"` freezes
    # everything here, which is what "frozen text encoder" means -- and this is
    # the same object n06 encodes the corpus with, so t_i, e_ij and the
    # retrieval space are one text understanding rather than three that happen
    # to share a name. Tokenisation is open_clip's 77-token context, applied by
    # `ULIPBackbone.encode_text`; truncation past 77 is CLIP's documented
    # behaviour and applies equally to every sentence.
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    backbone = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    out = []
    with torch.no_grad():
        for start in range(0, len(sentences), ENCODE_BATCH):
            emb = backbone.encode_text(sentences[start : start + ENCODE_BATCH])
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
    embeddings = np.concatenate(out, axis=0)
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # F8 makes e a load-bearing number, so a model swap that silently changed it
    # would change how much geometry survives message passing without anything
    # saying so.
    if embeddings.shape[1] != EDGE_DIM:
        raise ValueError(
            f"{TEXT_ENCODER} produced {embeddings.shape[1]}-d embeddings, not "
            f"{EDGE_DIM}. e_ij's width is a recorded decision (U-06, F8), not "
            "whatever the encoder happens to emit -- update EDGE_DIM "
            "deliberately or use the declared model."
        )
    return embeddings


def _write_json(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


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

    # [U-20] `procthor_node_embeddings`: ESSGNN's t_i, encoded HERE because this
    # is the node that already holds the encoder e_ij uses. Encoding them
    # elsewhere would make "the same encoder" a claim rather than a fact -- and
    # the two vectors are concatenated inside f_h, so a mismatch would put node
    # and edge features in unrelated spaces with nothing to signal it.
    node_texts = {a: rec["text"] for a, rec in text_map.items()}
    node_ids = sorted(node_texts)
    node_vecs = encode_sentences([node_texts[a] for a in node_ids])
    tmp_np = NODE_EMB_PATH.with_suffix(".part.npz")
    np.savez_compressed(tmp_np, ids=np.array(node_ids),
                        embeddings=node_vecs.astype(np.float32))
    tmp_np.replace(NODE_EMB_PATH)
    _write_json(NODE_EMB_RECORD, {
        "uri": str(NODE_EMB_PATH),
        "sha256": __import__("hashlib").sha256(NODE_EMB_PATH.read_bytes()).hexdigest(),
        "asset_ids": node_ids,
        "embedding_dim": int(node_vecs.shape[1]),
        "text_encoder_version": TEXT_ENCODER_VERSION,
        "n_assets": len(node_ids),
    })
    print(f"{len(node_ids):,} node embeddings (t_i) at {node_vecs.shape[1]}-d "
          f"-> {NODE_EMB_PATH}", flush=True)

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
