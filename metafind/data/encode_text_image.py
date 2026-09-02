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
import functools
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from metafind import paths, runlog
from metafind.data.render_blender import N_VIEWS as LIVE_N_VIEWS
from metafind.data.view_io import VIEW_IO_VERSION, image_identity

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
ENCODER_VERSION = 2   # v2 binds the sidecar to the checkpoint that produced it


@functools.lru_cache(maxsize=1)
def ulip2_ckpt_sha() -> str:
    """sha256 of the ULIP-2 checkpoint the encoder is built from.

    `docs/graph/node_registry.yaml:328` specifies this node's cache key as
    [annotation_sha256, render_sha256, ulip2_ckpt_sha]. `is_complete` compared
    the first two and nothing about the checkpoint, so swapping weights without
    bumping ENCODER_VERSION would resume across two encoders and every affected
    sidecar would pass forever -- a gallery in two halves, self-consistent and
    wrong, exactly the failure the text comparison was added to stop.

    It is worth binding even though today's checkpoint contributes only the
    point encoder (the open_clip half comes from `create_model_and_transforms`,
    `ulip_backbone.py:205`): "today's checkpoint does not touch text and image"
    is a fact about this file's contents, not a guarantee about the next one,
    and the sidecar cannot tell the difference after the fact.

    0.37 GiB, measured 0.4 s, once per process.
    """
    h = hashlib.sha256()
    with open(paths.ULIP2_CKPT, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()

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


def _retire(art: Path) -> Path:
    """Rename an artifact out of the way without destroying an earlier one.

    `art.replace(art.with_suffix(art.suffix + ".stale"))` OVERWRITES, and it
    does so on the exact path this code was written for: re-annotate, retire,
    re-annotate again, and the first `.stale` -- the evidence about the first
    failure -- is gone. Retiring is done instead of deleting because the file is
    evidence; silently destroying the older evidence defeats the reason.

    Suffixes are numbered rather than timestamped so the order is readable and
    the result does not depend on a clock.
    """
    target = art.with_suffix(art.suffix + ".stale")
    n = 1
    while target.exists():
        n += 1
        target = art.with_suffix(f"{art.suffix}.stale.{n}")
    art.replace(target)
    return target


def is_complete(uid: str, expected_text: str, image_id: str,
                aggregation: str | None = None,
                ckpt_sha: str | None = None) -> bool:
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
    # [ADDED 2026-08-24] Same rule as the text, applied to the images. An empty
    # id means n04's record could not say what it rendered, which is a reason to
    # re-encode, not a reason to accept.
    if not image_id or rec.get("image_identity") != image_id:
        return False
    # [ADDED 2026-08-24, Codex CHANGES REQUIRED] `image` IS the aggregate, so
    # the rule that produced it is part of this artifact's identity. The
    # protocol admits mean / max / fixed_view; without this, switching it
    # scheduled no work at all and Stage 1 trained on the old pooled vector
    # under the new label. (Stage 1 ALSO ignores its own aggregation argument --
    # `train/stage1.py` Stage1Dataset -- which is a separate node and is
    # reported, not fixed here.)
    if aggregation is not None and rec.get("aggregation") != aggregation:
        return False
    # [registry:328] The checkpoint half of the cache key. `None` means the
    # caller did not supply one, and that is a real case -- the tests construct
    # sidecars without a backbone -- so it cannot be required unconditionally.
    # What CAN be required is that a sidecar written by this version carries the
    # field at all: a v2 record with no `ulip2_ckpt_sha` was not written by this
    # code, whatever it claims about `encoder_version`.
    if "ulip2_ckpt_sha" not in rec:
        return False
    if ckpt_sha is not None and rec.get("ulip2_ckpt_sha") != ckpt_sha:
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
    if not npz.is_file():
        return False
    # [ADDED 2026-08-24, Codex CHANGES REQUIRED] Existence is not content. This
    # returned True for a file holding only `text`, for a truncated file, and
    # for arrays of the wrong width -- and n10's dataloader is where that
    # surfaced, mid-epoch, with nothing pointing back here. Worse: a
    # plausible-but-wrong width trains without any error at all.
    # The width comes from the RECORD, not from a constant imported here: the
    # sidecar states the dimension it wrote, so this checks the file against its
    # own claim and needs no dependency on the backbone module.
    dim = rec.get("embedding_dim")
    n_views = rec.get("n_views")
    try:
        with np.load(npz) as z:
            if not {"text", "views", "image"} <= set(z.files):
                return False
            if not isinstance(dim, int) or dim <= 0:
                return False
            if z["text"].shape != (dim,) or z["image"].shape != (dim,):
                return False
            if z["views"].ndim != 2 or z["views"].shape[1] != dim:
                return False
            # [FIXED 2026-08-24, Codex] `isinstance(n_views, int) and ...` made a
            # MISSING or string-valued `n_views` skip the row check entirely --
            # a validator that a malformed record disarms is not a validator.
            if not isinstance(n_views, int) or z["views"].shape[0] != n_views:
                return False
    except Exception:  # noqa: BLE001 -- unreadable is incomplete, whatever the reason
        return False
    return True


def _annotation_image_identity(annotation_path: Path) -> str:
    """The render identity n05 stamped on this annotation, or "" if it has none.

    "" is what every record written before 2026-08-24 carries, and it can never
    equal a real identity -- so those are excluded rather than silently joined to
    whatever is on disk now.
    """
    try:
        return json.loads(annotation_path.read_text()).get("image_identity") or ""
    except (OSError, json.JSONDecodeError):
        return ""


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
    # [ULIP2 REVIEWER MAJOR 1, 2026-09-03] The same treatment for the view
    # rule, and for the same reason.
    #
    # `view_aggregation` has seven fields and only `n_views` had a consumer:
    # the trainer's sidecar guard. The other six -- `selected_view_ids`,
    # `view_selection_policy`, `pre_normalize_each_view`, `method`,
    # `post_normalize`, `aggregation_version` -- were declared, folded into the
    # arm hash, and honoured by nothing. Setting `POST_NORMALIZE = True` was a
    # one-line edit that changed every recorded recipe, changed every arm hash,
    # changed nothing about `aggregate()` below (an unconditional
    # `views.mean(axis=0)`), and raised nothing. That is the "recorded recipe
    # that did not happen" defect this file has already been fixed for three
    # times, and the reviewer's verdict on the labels was right: they were
    # accurate, and a correct label is not a control.
    #
    # Comparing the artifact against what the code would produce NOW makes the
    # whole block honest at once. The comparison is total, not per field, so a
    # field added later inherits the check instead of needing to be remembered.
    from metafind.models.resolve_stage1 import view_aggregation

    expected_va = view_aggregation()
    if protocol.get("view_aggregation") != expected_va:
        raise ValueError(
            f"stage1_encoding_protocol records view_aggregation "
            f"{protocol.get('view_aggregation')!r}, but this code would produce "
            f"{expected_va!r}. Every field in that block describes what "
            "`aggregate()` actually does, so a difference means either the "
            "artifact was hand-edited or the code changed without re-resolving. "
            "Re-run n05b_resolve_stage1_encoding; do not edit the artifact."
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
        # [FIXED 2026-08-24] Was `Image.open(p).convert("RGB")`, which DROPS
        # alpha instead of compositing it. `view_io` exists precisely so that
        # the annotator and the encoder cannot disagree about what the model
        # saw, and this node -- named in that module's SUPPORTS-NODE line --
        # was the one call site still bypassing it. Every anti-aliased
        # silhouette edge in the corpus was a different colour here than in
        # n05: a 50%-alpha white edge came out 255 rather than 128.
        from metafind.data.view_io import load_views_rgb

        batch = self.torch.stack([self.preprocess(im)
                                  for im in load_views_rgb(view_paths)])
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
    # The WHOLE record, not only `view_paths`: `image_identity` needs the
    # renderer version and the per-view digests that live beside them.
    renders = {}
    for line in renders_index.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            renders[r["uid"]] = r

    paths.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    annotations = sorted(paths.ANNOTATIONS.glob("*.json"))
    # [B-1] Completion is decided against the text this serializer produces now,
    # so every sidecar written by a different serializer re-encodes. --force
    # still bypasses it, which is the one place a human states that intent.
    # [ADDED 2026-08-24, Codex CHANGES REQUIRED] An embedding joins TEXT from the
    # annotation to PIXELS from the renders. n05 stamps which images it was shown;
    # this node used to ignore that field entirely, so an annotation written
    # against renders that have since been replaced would be serialized as-is and
    # cached beside vectors from the NEW pixels -- a permanent stale-text /
    # new-image pair that then passes `is_complete()` forever. Re-annotating is
    # n05's job, so these are excluded and COUNTED rather than encoded or hidden.
    # An empty identity on EITHER side is a record that cannot say what it saw.
    # `"" == ""` used to pass this filter and get encoded, and then n06's own
    # completion check refused the same asset -- so it re-encoded on every resume
    # and never settled. Unknown is not a match.
    stale_text = [p for p in annotations
                  if p.stem in renders
                  and (not image_identity(renders[p.stem])
                       or _annotation_image_identity(p) != image_identity(renders[p.stem]))]
    stale = {p.stem for p in stale_text}
    # [ADDED 2026-08-24, Codex] Excluding the annotation is not enough: if an OLD
    # `<uid>.npz` is still there, n09 admits the uid and n10/n11 load that file
    # directly, so the stale embedding survives being "excluded". n06 declares
    # `partial_failure_semantics: halt`, so the artifact is RETIRED (renamed, not
    # deleted -- it is evidence) rather than left to be picked up downstream.
    for path in stale_text:
        for art in (paths.EMBEDDINGS / f"{path.stem}.npz",
                    sidecar_path(path.stem)):
            if art.exists():
                _retire(art)
    if stale_text:
        print(f"{len(stale_text):,} annotation(s) describe a different render than the "
              f"one on disk, e.g. {[p.stem for p in stale_text[:3]]}.\n"
              "Not encoding them: the text and the pixels would not be the same asset. "
              "Re-run n05 for these uids.", flush=True)
        # [N-1, 2026-08-24] HALT, unconditionally, and this used to be
        # `return 3 if stale_text else 0` guarded by `if not todo` twelve lines
        # below. That reached rc 3 only when the stale set was the WHOLE corpus.
        # With 500 stale and 45,500 fresh, `todo` was non-empty, the 500 were
        # excluded, their artifacts retired, and `main` returned **0** -- the
        # chain read success while the corpus had silently shrunk. The only
        # thing downstream of that was `chain_to_stage1.sh:71`'s
        # `[ "$EMB" -ge 45000 ]`, which admits losing ~955 assets without a word.
        #
        # `node_registry.yaml:334` (rank 5, above this code) declares n06
        # `partial_failure_semantics: halt`. The comment above already cited
        # that clause to justify RETIRING the artifacts, and then did not halt
        # -- a citation that licenses the convenient half of a rule is worse
        # than no citation, because it is what stops the next reader checking.
        #
        # Retire first, then halt: the retirement must survive the halt or a
        # resume would find the stale `.npz` still in place and admit it.
        return 3
    todo = [p for p in annotations
            if p.stem in renders and p.stem not in stale
            and (args.force or not is_complete(p.stem, expected_text_for(p),
                                               image_identity(renders[p.stem]),
                                               protocol["image_aggregation"],
                                               ulip2_ckpt_sha()))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(annotations):,} annotated, {len(todo):,} to encode "
          f"(aggregation: {protocol['image_aggregation']})", flush=True)
    if not todo:
        # `stale_text` can no longer be non-empty here -- the halt above returns
        # 3 before this point -- so nothing left to encode now means exactly
        # that: the corpus is complete.
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
                view_vecs = enc.encode_views(renders[uid]["view_paths"])
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
                "ulip2_ckpt_sha": ulip2_ckpt_sha(),
                "embedding_uri": str(npz),
                "text": text,
                "text_tokens": n_tokens,
                "text_truncated": truncated,
                "n_views": int(view_vecs.shape[0]),
                "aggregation": protocol["image_aggregation"],
                "embedding_dim": int(text_vec.shape[0]),
                "text_serialization": protocol["text_serialization"],
                "clip_train_scope": protocol["actual_clip_train_scope"],
                # What these image vectors were taken from. See `image_identity`.
                "image_identity": image_identity(renders[uid]),
                "view_io_version": VIEW_IO_VERSION,
                "renderer_version": renders[uid].get("renderer_version"),
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
                       # [FIXED 2026-08-24] Was `done * 11`; the live artifact
                       # carries 12 views. Read it off the renderer rather than
                       # restating a literal that a version bump can strand.
                       assets_encoded=done, views_encoded=done * LIVE_N_VIEWS)
    print(f"\n{done:,} encoded, {overlong:,} REFUSED for exceeding "
          f"{TEXT_CONTEXT_LENGTH} true tokens -> {paths.EMBEDDINGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
