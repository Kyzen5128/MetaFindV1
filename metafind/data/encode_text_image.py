"""Cache the frozen CLIP halves of every asset's embedding.

# IMPLEMENTS-NODE: n06_encode_text_image

Writes ``text_image_embeddings`` (one sidecar per asset), ``run_progress`` and
``cost_ledger``.

What may be cached, and what may not
------------------------------------

Only text and image, and only while ``actual_clip_train_scope == frozen``.

A cache of an encoder's output is valid exactly as long as that encoder is not
being updated. The point cloud fails that test: paper 2.6 trains the point
encoder in Stage 1, so a cached point embedding would be the output of a network
that is about to change. An earlier draft cached all three, which turned the main
line into Table 3's "train fuser only" row -- the row the paper reports as WORSE
(8.7 against 11.4). The same reasoning binds CLIP, which is why this node reads
the protocol rather than assuming: under `trainable` it does not run at all, and
n10 consumes renders and annotations directly.

Why all eleven view embeddings are stored, not one
---------------------------------------------------

[U-14] n05b resolved the aggregation to `mean`. Storing the aggregate alone
would bake that decision into 46,052 files, so a later variant -- and Table 1's
image conditions are exactly where aggregation matters -- would need a full
re-encode to answer a question the cache could have answered. Eleven 1280-d
float16 vectors cost 28 KB per asset; the whole corpus is about 1.3 GB.

The aggregation still travels with the record, because L1-IMAGE-AGGREGATION
requires the rule that WAS applied to be recorded beside the embedding rather
than inferred from a config file later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog

# Before transformers or open_clip reaches this process: HF_HOME is read at
# import time, and ViT-bigG-14 is 9.5 GB. n05 learned this the slow way.
paths.setup_env()

from metafind.models.resolve_stage1 import (  # noqa: E402
    TEXT_TEMPLATE,
    serialization_id_for,
    serialize_annotation,
)
from metafind.models.stage1_config import (  # noqa: E402
    PRECOMPUTABLE_AGGREGATIONS,
)

NODE = "n06_encode_text_image"
ENCODER_VERSION = 1

# CLIP truncates at 77 tokens silently, and the pinned template puts the
# description first -- so an overlong one loses the TAIL, where the placement
# constraint lives. n05b caps by CHARACTERS, which bounds the realistic case;
# this is L1-TEXT-TOKEN-BUDGET, the bound on the rest.
#
# [P-4] An overflow is now REFUSED, not merely recorded. The previous code set
# `text_truncated=True` and encoded the asset anyway, so a knowingly-degraded
# embedding entered the gallery with a flag nothing downstream reads. It also
# counted with the PADDED tokenizer, which saturates at exactly 77 -- the
# corpus's 89-token record reported 77 and looked like a boundary case rather
# than a 12-token loss.
TEXT_CONTEXT_LENGTH = 77


def true_token_count(text: str) -> int:
    """SOT + BPE + EOT, untruncated and unpadded.

    NOT `open_clip.tokenize`, and not `Encoder.token_count`, both of which pad to
    exactly TEXT_CONTEXT_LENGTH and cannot report a number larger than it. The
    tokenizer is BPE only -- no model weights, no GPU -- so this is also what the
    pre-flight gate uses before any encoder exists.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from open_clip.tokenizer import SimpleTokenizer

        _TOKENIZER = SimpleTokenizer()
    return len(_TOKENIZER.encode(text)) + 2


_TOKENIZER = None


def refuse_if_overlong(text: str) -> int:
    """[L1-TEXT-TOKEN-BUDGET, P-4] The gate, as a function, so it can be tested.

    Returns the true token count, or raises. Called per asset inside the encode
    loop, where the raise lands the asset in quarantine with the count attached.
    """
    n_tokens = true_token_count(text)
    if n_tokens > TEXT_CONTEXT_LENGTH:
        raise ValueError(
            f"serialized text is {n_tokens} true BPE tokens, over CLIP's "
            f"{TEXT_CONTEXT_LENGTH}-token context. Encoding it would silently "
            "drop the tail, which is the placement clause. Run "
            "tools/preflight_stage1_text.py and repair the annotation before "
            "encoding."
        )
    return n_tokens


def sidecar_path(uid: str) -> Path:
    return paths.EMBEDDINGS / f"{uid}.json"


def is_complete(uid: str, expected_text: str) -> bool:
    """[B-1, D0-008 §11.2] Complete means: this sidecar holds the text THIS
    serializer produces for THIS uid, and the vectors it points at exist.

    The earlier version compared nothing about the text -- only sidecar
    existence, ``encoder_version`` and NPZ existence. That made a resumed run
    skip all 5,276 metre-template embeddings as "complete" and encode the rest
    under the centimetre template: a gallery built from two text distributions,
    with no error, no warning, and the same ``text_serialization`` label on both
    halves. ``gallery_index.py`` fingerprints the checkpoint rather than the
    text, so Table 1 would have come out self-consistent and wrong.

    Binding to the exact string is stronger than any version counter, because
    it also catches a re-annotated record whose serializer never changed. Its
    cost is one small JSON read per asset in the work-list pass -- seconds
    against ~4 GPU-hours.

    ``expected_text`` is empty for a record this serializer cannot serialize at
    all (the 3 ``prompt_version:1`` residuals raise ``KeyError: 'width'``).
    Those are incomplete here and are quarantined by the encode loop, which is
    where they were already handled.
    """
    if not expected_text:
        return False
    sc = sidecar_path(uid)
    if not sc.exists():
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if rec.get("encoder_version") != ENCODER_VERSION:
        return False
    if rec.get("text") != expected_text:
        return False
    # Not `Path(rec.get("embedding_uri", "")).exists()`. An absent or empty field
    # made that `Path(".")`, the working directory, which exists -- so a sidecar
    # with no vectors at all was judged complete and its asset was skipped. The
    # failure then surfaced as a FileNotFoundError inside n10's dataloader,
    # mid-epoch, with nothing pointing back at n06.
    uri = rec.get("embedding_uri")
    if not isinstance(uri, str) or not uri:
        return False
    # It must be THIS asset's vectors, not merely some existing file. Requiring
    # only is_file() let a sidecar point at any regular path in the repository
    # and still be judged complete; requiring only exists() was worse, because
    # Path("") is Path(".") and a directory exists.
    npz = paths.EMBEDDINGS / f"{uid}.npz"
    try:
        if Path(uri).resolve() != npz.resolve():
            return False
    except OSError:
        return False
    return npz.is_file()


def expected_text_for(annotation_path: Path) -> str:
    """The string this serializer would produce, or ``""`` if it cannot.

    A record that fails to serialize is not "complete"; it is quarantined by the
    encode loop with the real traceback. Swallowing the exception here would only
    hide which record it was, so nothing is logged at this point.
    """
    try:
        return serialize_annotation(json.loads(annotation_path.read_text()))
    except Exception:  # noqa: BLE001 -- the encode loop raises it again and quarantines
        return ""


def load_protocol() -> dict:
    path = paths.OUTPUTS / "stage1_encoding_protocol.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- n05b_resolve_stage1_encoding must run first. "
            "Encoding before the protocol exists is what n05b was split out to "
            "prevent."
        )
    protocol = json.loads(path.read_text())
    if protocol.get("status") != "resolved":
        raise ValueError(f"stage1_encoding_protocol is {protocol.get('status')!r}")
    if protocol["actual_clip_train_scope"] != "frozen":
        raise ValueError(
            "actual_clip_train_scope is 'trainable', so nothing may be cached: "
            "gradient has to reach OpenCLIP and a cache is the output of a "
            "network that is being updated. n06 does not run under that reading."
        )
    if protocol["image_aggregation"] not in PRECOMPUTABLE_AGGREGATIONS:
        raise ValueError(
            f"image_aggregation {protocol['image_aggregation']!r} is per-view, "
            "so a single cached vector cannot honour it"
        )
    # [B-2, D0-008 §11.2] The recorded protocol must describe the serializer this
    # process actually imported. Until this check existed the artifact recorded
    # the metre-based v1 template while the code emitted the centimetre one, and
    # line 233 stamped the artifact's label on embeddings the code had produced:
    # the system could encode with serializer X and label it Y without failing.
    # An unenforced record is decorative.
    # Derived through this module's OWN alias, not through the resolver's. The
    # two are normally the same function; if anything rebinds one of them, the
    # protocol must certify the callable that will actually serialize, not the
    # one the resolver happens to hold.
    expected_id = serialization_id_for(serialize_annotation)
    if protocol.get("text_serialization") != expected_id:
        raise ValueError(
            f"stage1_encoding_protocol records text_serialization "
            f"{protocol.get('text_serialization')!r}, but this process's "
            f"serializer is {expected_id!r}. The identity is a hash of the "
            "string serialize_annotation() actually emits, so these differ only "
            "if the protocol was written by a different serializer. Re-run "
            "n05b_resolve_stage1_encoding; do not edit the artifact by hand."
        )
    if protocol.get("text_template") != TEXT_TEMPLATE:
        raise ValueError(
            "stage1_encoding_protocol records a text_template that is not the "
            "one this process imported. The identity above already matched, so "
            "the artifact's template field has been edited independently of the "
            "code that produced it. Re-run n05b_resolve_stage1_encoding."
        )
    return protocol


class Encoder:
    """The frozen ViT-bigG-14 halves. One instance per process."""

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

        self.torch = torch
        # train_scope is fuser_only so nothing here builds an autograd graph:
        # this node exists precisely because these weights do not move.
        self.backbone = ULIPBackbone(BackboneConfig(device=device,
                                                    train_scope="fuser_only"))
        self.preprocess = self.backbone.preprocess

    def encode_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            out = self.backbone.encode_text([text])
        return out[0].float().cpu().numpy()

    def encode_views(self, view_paths: list[str]) -> np.ndarray:
        from PIL import Image

        batch = self.torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                  for p in view_paths])
        with self.torch.no_grad():
            out = self.backbone.encode_image(batch)
        return out.float().cpu().numpy()

    def token_count(self, text: str) -> int:
        return int((self.backbone.tokenizer([text])[0] != 0).sum())


def aggregate(views: np.ndarray, rule: str) -> np.ndarray:
    """[U-14, L1-IMAGE-AGGREGATION] The rule n05b resolved, applied here.

    `mean` on the raw embeddings rather than on L2-normalised ones: the tower
    normalises at comparison time, and normalising twice would weight every view
    equally regardless of how confidently the encoder placed it.
    """
    if rule == "mean":
        return views.mean(axis=0)
    if rule == "max":
        return views.max(axis=0)
    if rule == "fixed_view":
        return views[0]
    raise ValueError(f"{rule!r} is not a precomputable aggregation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    protocol = load_protocol()
    renders_index = paths.LOGS / "renders_index.jsonl"
    if not renders_index.exists():
        print(f"{renders_index} not found -- run n04 first", flush=True)
        return 2
    renders = {}
    for line in renders_index.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            renders[r["uid"]] = r["view_paths"]

    paths.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    annotations = sorted(paths.ANNOTATIONS.glob("*.json"))
    # [B-1] Completion is decided against the text this serializer produces now,
    # so every sidecar written by a different serializer re-encodes. --force
    # still bypasses it, which is the one place a human states that intent.
    todo = [p for p in annotations
            if p.stem in renders
            and (args.force or not is_complete(p.stem, expected_text_for(p)))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(annotations):,} annotated, {len(todo):,} to encode "
          f"(aggregation: {protocol['image_aggregation']})", flush=True)
    if not todo:
        return 0

    enc = Encoder()
    done, overlong, started = 0, 0, time.time()
    with runlog.run_progress(NODE):
        for path in todo:
            uid = path.stem
            try:
                annotation = json.loads(path.read_text())
                text = serialize_annotation(annotation)
                # [L1-TEXT-TOKEN-BUDGET, P-4] Counted on the untruncated path
                # and ENFORCED. The raise sends the asset to quarantine with the
                # count attached, which is the same treatment an unserializable
                # annotation gets -- a degraded input must be visible, not quiet.
                try:
                    n_tokens = refuse_if_overlong(text)
                except ValueError:
                    overlong += 1
                    raise
                truncated = False

                text_vec = enc.encode_text(text)
                view_vecs = enc.encode_views(renders[uid])
                pooled = aggregate(view_vecs, protocol["image_aggregation"])

                npz = paths.EMBEDDINGS / f"{uid}.npz"
                tmp = paths.EMBEDDINGS / f"{uid}.part.npz"
                np.savez_compressed(tmp,
                                    text=text_vec.astype(np.float16),
                                    views=view_vecs.astype(np.float16),
                                    image=pooled.astype(np.float16))
                tmp.replace(npz)
            except Exception as exc:  # noqa: BLE001 -- one asset must not stop the run
                runlog.quarantine(NODE, [{
                    "uid": uid,
                    "failure_class": "DETERMINISTIC_INPUT",
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-1500:],
                }])
                continue

            rec = {
                "uid": uid,
                "encoder_version": ENCODER_VERSION,
                "embedding_uri": str(npz),
                "text": text,
                "text_tokens": n_tokens,
                "text_truncated": truncated,
                "n_views": int(view_vecs.shape[0]),
                "aggregation": protocol["image_aggregation"],
                "embedding_dim": int(text_vec.shape[0]),
                "text_serialization": protocol["text_serialization"],
                "clip_train_scope": protocol["actual_clip_train_scope"],
            }
            sc = sidecar_path(uid)
            tmp = sc.with_suffix(".json.part")
            with tmp.open("w") as fh:
                json.dump(rec, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(sc)
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                print(f"  [{done:6d}/{len(todo)}] {rate:.0f}/min, "
                      f"over-length text {overlong}", flush=True)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       assets_encoded=done, views_encoded=done * 11)
    print(f"\n{done:,} encoded, {overlong:,} REFUSED for exceeding "
          f"{TEXT_CONTEXT_LENGTH} true tokens -> {paths.EMBEDDINGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
