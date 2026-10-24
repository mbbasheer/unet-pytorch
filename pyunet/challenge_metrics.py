"""Python versions of the ISBI 2012 challenge's pixel error and Rand error.

They follow ISBI-2012-challenge/challenge-error-metrics.bsh: the proposal is a
P(cell) map, it is thresholded at 0.0, 0.1, ..., 1.0, and the best F-score over
those thresholds is kept (error = 1 - best F-score). The challenge's own code
lives in Fiji (trainableSegmentation.metrics), so details such as pixel
connectivity may differ slightly; treat these as close approximations.

The warping error is not implemented here.
"""

import numpy as np
from scipy import ndimage

THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def pixel_error(probs, labels, thresholds=THRESHOLDS):
    """1 - maximal F-score of pixel similarity, with cell (label 1) as the positive class.

    Returns (error, best_threshold). Counts are pooled over all slices.
    """
    truth = labels == 1
    best_f, best_t = 0.0, None
    for t in thresholds:
        pred = probs > t
        tp = np.count_nonzero(pred & truth)
        fp = np.count_nonzero(pred & ~truth)
        fn = np.count_nonzero(~pred & truth)
        f = 2 * tp / (2 * tp + fp + fn)
        if f > best_f:
            best_f, best_t = f, float(t)
    return 1.0 - best_f, best_t


def pixel_mismatch(probs, labels, thresholds=THRESHOLDS):
    """Lowest fraction of wrongly classified pixels over the thresholds (the older definition)."""
    truth = labels == 1
    errors = [np.mean((probs > t) != truth) for t in thresholds]
    i = int(np.argmin(errors))
    return float(errors[i]), float(thresholds[i])


def _rand_sums(pred_cells, true_cells):
    """Pair-count sums for the foreground-restricted Rand index of one slice.

    Only pixels inside a ground-truth cell count. Predicted membrane pixels there
    (label 0) are treated as single-pixel segments.
    """
    mask = true_cells > 0
    p = pred_cells[mask].astype(np.int64)
    t = true_cells[mask].astype(np.int64)
    singletons = np.count_nonzero(p == 0)
    p_fg, t_fg = p[p > 0], t[p > 0]
    joint = np.bincount(p_fg * (t.max() + 1) + t_fg).astype(np.float64)
    sum_joint = (joint**2).sum() + singletons
    sum_pred = (np.bincount(p_fg).astype(np.float64) ** 2).sum() + singletons
    sum_true = (np.bincount(t).astype(np.float64) ** 2).sum()
    return sum_joint, sum_pred, sum_true


def rand_error(probs, labels, thresholds=THRESHOLDS):
    """1 - maximal F-score of the foreground-restricted Rand index.

    Cells are 4-connected regions: of the ground truth's cell pixels, and of the
    proposal's pixels above the threshold. Returns (error, best_threshold).
    """
    true_cells = [ndimage.label(label == 1)[0] for label in labels]
    best_f, best_t = 0.0, None
    for t in thresholds:
        totals = np.zeros(3)
        for prob, truth in zip(probs, true_cells):
            pred_cells = ndimage.label(prob > t)[0]
            totals += _rand_sums(pred_cells, truth)
        sum_joint, sum_pred, sum_true = totals
        precision, recall = sum_joint / sum_pred, sum_joint / sum_true
        f = 2 * precision * recall / (precision + recall)
        if f > best_f:
            best_f, best_t = f, float(t)
    return 1.0 - best_f, best_t
