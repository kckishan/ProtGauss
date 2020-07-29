# borrowed from https://github.com/VGligorijevic/deepNF/blob/master/validation.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


def real_AUPR(label, score):
    """Computing real AUPR . By Vlad and Meet"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0

    return pr, f


def ml_split(y, num_splits=10, seed=0):
    """Split annotations"""
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def evaluate_performance(y_test, y_score, y_pred, alpha):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    if n == 0:
        n = 1
    perf["M-aupr"] /= n

    # Compute micro-averaged AUPR
    pr, _ = real_AUPR(y_test, y_score) 
    perf["m-aupr"] = pr if pr == pr else 0

    # Computes accuracy
    perf['acc'] = accuracy_score(y_test, y_pred)

    # Computes F1-score
    y_new_pred = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')

    return perf
