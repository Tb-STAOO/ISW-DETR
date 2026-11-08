"""Functions preserved from the original experiment implementation."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_matrix(boxes, ground):
    low = np.maximum(boxes[:, None, :2], ground[None, :, :2])
    high = np.minimum(boxes[:, None, 2:], ground[None, :, 2:])
    overlap = np.maximum(high-low, 0).prod(-1)
    a = np.maximum(boxes[:, 2:]-boxes[:, :2], 0).prod(-1)
    b = np.maximum(ground[:, 2:]-ground[:, :2], 0).prod(-1)
    return overlap / np.maximum(a[:, None]+b[None, :]-overlap, 1e-12)


def maximum_matching(ious, query_indices):
    # A cardinality bonus makes one extra valid pair dominate every IoU tie-break.
    if not len(query_indices) or not ious.shape[1]:
        return {}
    current = ious[query_indices]
    valid = current >= .5
    rows, columns = linear_sum_assignment(-(valid*(min(current.shape)+1)+current*valid))
    return {int(query_indices[r]): int(c) for r,c in zip(rows,columns) if valid[r,c]}


def score_order_matching(ious, logits):
    scores = 1/(1+np.exp(-np.clip(logits.max(-1),-80,80)))
    available = set(range(ious.shape[1]))
    matched = {}
    for q in np.argsort(-scores,kind='stable'):
        if scores[q] < .05 or not available:
            break
        g = max(sorted(available), key=lambda index: ious[q,index])
        if ious[q,g] >= .5:
            matched[int(q)] = int(g)
            available.remove(g)
    return matched

