# Upstream verbatim harvest — ULIP-1 and Point-BERT papers

Read word-for-word by Master, 2026-08-25, from the arXiv sources Kyzen downloaded.
Files: `docs/paper/ulip1_source/main.tex` (1,125 lines) · `docs/paper/pointbert_source/Pointbert_arxiv.tex` (624 lines).
Both carry `SOURCE_MANIFEST.json` with sha256. Every claim below is UPSTREAM FACT with a line citation.

---

## ULIP-1 (arXiv 2212.05171v4, CVPR 2023)

### The loss — this is what "ULIP-2 inherits ULIP's framework" resolves to

- `main.tex:267-273` — each pairwise contrastive loss is **symmetric**: ½ the (i→j) direction
  + ½ the (j→i) direction, InfoNCE form.
- `main.tex:279` — "We use a **learnable temperature** parameter τ as well, similar to CLIP."
- `main.tex:282-285` — **L_final = α·L(I,S) + β·L(I,P) + θ·L(P,S)** with
  "**α is set to be constant 0, β and θ are set to be 1**".
  So: image↔text loss is OFF; trained pairs are **3D↔image and 3D↔text only**, each bidirectional.
- **Consequence for MetaFind Stage 1**: MetaFind's own Eq. (2methdology.tex:76-78) is a
  *single-direction query→gallery* loss with fixed τ=0.5. Where MetaFind speaks it overrides
  upstream: single-direction Stage 1 + fixed τ are MetaFind PAPER FACTS. Upstream's
  bidirectional-pairwise form is what Stage 2's ½(q2g+g2q) resembles.

### The freezing — DL-032's ground, now with the upstream's own reason

- `main.tex:286` verbatim: "if we update CLIP's image and text encoders, **catastrophic
  forgetting will emerge due to our limited data size**. This will lead to a significant
  performance drop … Therefore we **freeze** the weights of f_S(·) and f_I(·) during the
  **entire pre-training** and only update f_P(·)".
- `main.tex:1078-1092` (commented-out ablation table left in the source): unfreezing CLIP →
  zero-shot top-1 **0.0** vs **37.1** frozen. Never published, but it is in the authors' own
  source. Freezing is not stylistic — unfreezing collapsed the model.

### Pretraining recipe — the citable origin of the numbers floating around

- `main.tex:367-370` verbatim: "we utilize an advanced version of CLIP, namely **SLIP** … we
  freeze the image and text encoders and only update the 3D encoder's parameters …
  ULIP is trained for **250 epochs**. We use **64 as the batch size**, **10⁻³ as the learning
  rate**, and **AdamW** as the optimizer."
- `main.tex:383` — pretraining on **8 A100**, finetuning on 1.
- So: lr 1e-3 / batch 64 / AdamW = **ULIP-1 PAPER**. The consultant's attribution of these to
  the *ULIP-2* paper stays wrong (no ULIP-2 version contains them); "50 epochs Objaverse"
  remains UNVERIFIED in any paper we hold.
- Note: ULIP-1 uses **SLIP**; ULIP-2 switched to **OpenCLIP ViT-G/14** (ulip2 main.tex:609).
  Ours follows ULIP-2 (ViT-bigG-14).

### Data construction

- `main.tex:218` — 3D input: uniform sample Np ∈ {1024, 2048, 8192} per backbone; train-time
  cloud augmentation: random point drop, scale, shift, rotate perturbation.
- `main.tex:236` — renders: 30 RGB + 30 depth per object (one per 12°);
  **each iteration randomly selects ONE of the 60** as the image input.
- `main.tex:244-250` — text: 63 standard prompts + "a point cloud model of [WORD]" = **64
  templates**, encoded and **average-pooled** into one text feature.

---

## Point-BERT (arXiv 2111.14819v2, CVPR 2022) — our only trainable encoder

### Architecture (what our vendored `pointbert` implements)

- `:121` — input pipeline: **FPS g centers → kNN n neighbours per patch → subtract centre**
  (disentangle pattern from position) → **mini-PointNet (2-layer MLP)** patch embeddings.
- `:141` — positional embedding = MLP(centre coords); **[CLS] token appended**; L Transformer
  blocks; output = CLS + per-patch tokens.
- `:216/:594` — **depth 12, dim 384, heads 6, stochastic depth 0.1**.
- `:597` — classification head input = **Concat(CLS, max-pool over patch tokens)** → 2-layer MLP.
  (Matches how ULIP/our backbone reads the global feature.)
- Paper-scale grouping: 1024 points → 64 patches × 32 (`:174`). **ULIP-2's colored variant
  scales this to npoints 10000 / num_group 512 / group_size 32**
  (`upstream/ULIP/models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml`) — that yaml,
  not this paper, is the citation for our input scale.

### Pretraining machinery (dVAE / MPM / MoCo)

- Relevant only to how the released Point-BERT weights were made; none of it runs in MetaFind.
  dVAE tokenizer (DGCNN, vocab 8192), block-mask MPM [0.25–0.45], MoCo memory 16,384 τ=0.07;
  AdamW 5e-4 / wd 0.05 / 300 epochs / batch 128 (`:214-216`, appendix tables).

---

## Still unread from this batch (queued)

`openshape_source` (672 tex lines) · `procthor_source` (2,417) · `flamingo_source` (3,593).
OpenShape next (render/retrieval upstream); ProcTHOR before ESSGNN opens; Flamingo feeds the
λ-init debate only.
