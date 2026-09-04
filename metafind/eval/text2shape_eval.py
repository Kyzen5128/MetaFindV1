"""Text2Shape's retrieval metrics, copied VERBATIM, run beside ours.

[KYZEN 2026-09-04, verbatim] 「她明明就有 … 那你不照抄」-- upstream ships a
retrieval evaluation; use it rather than our own reading of "R@k".

UPSTREAM FACT, Text2Shape (Chen et al., arXiv 1803.08495v1, §5.1): "We measure
retrieval performance using normalized discounted cumulative gain (NDCG) [45]
… and recall rate (RR@k) [12, 46], which considers a retrieval successful if at
least one sample in the top k retrievals is of the correct class." The "class"
is the shape a description belongs to ("retrieve descriptions and shapes
belonging to the same configuration … we only have ground truth association
for one shape"), i.e. exact-instance, one relevant item per query.

UPSTREAM FACT, official code `tools/eval/eval_text_encoder.py` (MIT, (c) 2018
Kevin Chen, clone at /home/kyzen/upstream/text2shape @ 2f62ebc):
  * `compute_nearest_neighbors` accepts ONLY metric='cosine' and that path is
    `np.dot(query, fit.T)` on the embeddings AS GIVEN -- the file itself prints
    "Using unnormalized cosine distance". So the shipped scorer is a raw dot
    product; it is cosine only if the encoder emits unit vectors.
  * `compute_pr_at_k` derives precision@k, recall@k, recall_rate@k (RR@k) and
    NDCG@k for k = 1..n_neighbors from the top-n_neighbors index list and the
    label of each row; `num_relevant` counts how many gallery rows share the
    query's label.
  * Top-k is `np.argpartition` on the similarities: TIES are broken by the
    partition's arbitrary order, not against the model as ours are.

The three functions below are copied unchanged from that file (comments and
dead branches included) so a reader can diff them against upstream. Only the
wrapper `text2shape_metrics` is ours: it feeds our (query, gallery, target)
triple through them with the gallery as `fit` and one label per gallery row.
"""
from __future__ import annotations

import collections
import contextlib
import io

import numpy as np


# --------------------------------------------------------------------------
# VERBATIM from text2shape/tools/eval/eval_text_encoder.py (MIT, Kevin Chen)
# --------------------------------------------------------------------------

def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix,
                                      n_neighbors, fit_eq_query, range_start=0):
    if fit_eq_query is True:
        n_neighbors += 1

    # print('Using unnormalized cosine distance')

    # Argsort method
    # unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    # sort_indices = np.argsort(unnormalized_similarities, axis=1)
    # # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    # indices = sort_indices[:, -n_neighbors:]
    # indices = np.flip(indices, 1)

    # Argpartition method
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    n_samples = unnormalized_similarities.shape[0]
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    indices = sort_indices[:, -n_neighbors:]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)

    if fit_eq_query is True:
        n_neighbors -= 1  # Undo the neighbor increment
        final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
        compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
        has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
        any_result = np.any(has_self, axis=1)
        for row_idx in range(indices.shape[0]):
            if any_result[row_idx]:
                nonzero_idx = np.nonzero(has_self[row_idx, :])
                assert len(nonzero_idx) == 1
                new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
                final_indices[row_idx, :] = new_row
            else:
                final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
        indices = final_indices
    return indices


def compute_pr_at_k(indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)

    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k from 0 to n_neighbors(k+1 from 1 to n_neighbors)
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    #print('recall_at_k shape:', recall_at_k.shape)
    print('     k: precision recall recall_rate ndcg')
    for k in range(n_neighbors):
        print('pr @ {}: {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k]))
    Metrics = collections.namedtuple('Metrics', 'precision recall recall_rate ndcg')
    return Metrics(precision_at_k, recall_at_k, recall_rate_at_k, ave_ndcg_at_k)


# --------------------------------------------------------------------------
# ours: the adapter
# --------------------------------------------------------------------------

N_NEIGHBORS = 5     # Text2Shape reports RR@1, RR@5, NDCG@5 (Table 1 of the paper)


def text2shape_metrics(query: np.ndarray, gallery: np.ndarray, targets: np.ndarray,
                       n_neighbors: int = N_NEIGHBORS) -> dict:
    """RR@1, RR@5, NDCG@5 (and precision@k) as Text2Shape computes them.

    `query` (n_q, D) are the queries, `gallery` (n_g, D) the fit matrix,
    `targets[i]` the gallery ROW holding query i's own asset. Labels are the
    gallery row index itself, so every gallery row is its own class and each
    query has exactly one relevant item -- Text2Shape's ShapeNet situation
    ("ground truth association for one shape").

    Pass the vectors you mean: unit vectors give cosine, raw tower outputs give
    the unnormalised dot product upstream actually runs. Both are reported by
    `run_retrieval`.
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    t = np.asarray(targets, dtype=np.int64)
    if q.shape[0] != t.shape[0]:
        raise ValueError(f"{q.shape[0]} queries but {t.shape[0]} targets")
    if g.shape[0] < n_neighbors:
        raise ValueError(f"gallery of {g.shape[0]} rows cannot yield top-{n_neighbors}")
    indices = _compute_nearest_neighbors_cosine(g, q, n_neighbors, fit_eq_query=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):       # upstream prints its table; keep it, quietly
        m = compute_pr_at_k(indices, t, n_neighbors, q.shape[0],
                            fit_labels=np.arange(g.shape[0]))
    return {"RR@1": float(m.recall_rate[0]),
            f"RR@{n_neighbors}": float(m.recall_rate[n_neighbors - 1]),
            f"NDCG@{n_neighbors}": float(m.ndcg[n_neighbors - 1]),
            "precision@1": float(m.precision[0]),
            "n_neighbors": int(n_neighbors),
            "upstream_table": buf.getvalue().strip().splitlines()}
