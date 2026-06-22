import numpy as np
import statistics
import csv
import os
from datetime import datetime
from uuid import uuid4

from .quantifiers_utils import getHist, getTPRandFPRbyThreshold, DySyn_distance, TernarySearch, MoSS

def write_distributions(moss_distributions, distributions_dir=None):
    if distributions_dir is None:
        distributions_dir = os.path.join(os.getcwd(), "distributions")
    os.makedirs(distributions_dir, exist_ok=True)
    file_name = f"dysyn_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.csv"
    file_path = os.path.join(distributions_dir, file_name)

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["p_scores", "n_scores", "n", "alpha", "mf"])
        writer.writeheader()
        writer.writerows(moss_distributions)

    return file_path

def CC(test, thr=0.5):
    result = np.sum(test >= thr) / len(test)
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def CC2(test):
    # Count which column has the maximum value for each row
    max_indices = np.argmax(test, axis=1)
    
    # Count occurrences of each column index
    num_columns = test.shape[1]
    result = np.zeros(num_columns)
    
    for idx in max_indices:
        result[idx] += 1
    
    # Normalize to get proportions
    result = result / len(test)
    
    return result.astype(float)

def ACC(test, TprFpr, thr=0.5):
    dC = CC(test)  # Implement CC function separately

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == thr, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def DyS(p_score, n_score, test, measure="topsoe", bins=np.arange(2, 22, 2), err=1e-5):
    results = []

    for b_size in bins:
        Sty_1 = getHist(p_score, b_size)  # Implement getHist separately
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)  # Implement DyS_distance separately

        best_alpha = TernarySearch(0, 1, f, err)  # Implement TernarySearch separately
        results.append(best_alpha)

    result = statistics.median(results)
    result = max(0, min(result, 1))

    return np.array([result, 1 - result])

def DySyn(ts, measure, MF=np.arange(0.1, 1.0, 0.2), write_distribution=None, distributions_dir=None, return_metadata=False):
    # MF = np.arange(0.2, 0.7, 0.2) # mudar de 0.2 a 0.7
    # Ensure MF is always iterable (handles scalar numpy.float64 / 0-d arrays)
    MF = np.atleast_1d(np.round(MF, 2)).astype(float)
    # pdb.set_trace()

    results = []
    distances = []
    moss_distributions = []
    synthetic_candidates = []
    n_samples = 1000
    alpha = 0.5

    for mf in MF:
        scores = MoSS(n_samples, alpha, mf)  # Implement MoSS function separately
        test_p = scores[scores[:, 2] == 1, 0]
        test_n = scores[scores[:, 2] == 2, 0]
        synthetic_candidates.append({
            "p_scores": test_p,
            "n_scores": test_n,
            "n": n_samples,
            "alpha": alpha,
            "mf": mf,
        })
        moss_distributions.append({
            "p_scores": np.array2string(test_p, separator=","),
            "n_scores": np.array2string(test_n, separator=","),
            "n": n_samples,
            "alpha": alpha,
            "mf": mf,
        })

        if measure == "sord":
            rQnt = DySyn_SORD(test_p, test_n, ts)  # Implement DySyn_SORD separately
        else:
            rQnt = DySyn_DyS(test_p, test_n, ts, measure, [10])  # Implement DySyn_DyS separately

        distances.append(rQnt[1])
        results.append(rQnt[0][0])

    distribution_file = None
    if write_distribution:
        distribution_file = write_distributions(moss_distributions, distributions_dir=distributions_dir)

    best_idx = int(np.argmin(distances))
    best_result = round(results[best_idx], 2)
    response = [np.array([best_result, 1 - best_result]), min(distances), MF[best_idx]]

    if return_metadata:
        response.append({
            "selected_p_scores": synthetic_candidates[best_idx]["p_scores"],
            "selected_n_scores": synthetic_candidates[best_idx]["n_scores"],
            "selected_mf": synthetic_candidates[best_idx]["mf"],
            "distribution_file": distribution_file,
        })

    return response

def DySyn_DyS(p_score, n_score, test, measure="hellinger", b_sizes = list(range(2, 21, 2)) + [30]):
    results = []
    vDistAll = []

    for b_size in b_sizes:
        Sty_1 = getHist(p_score, b_size)  # Implement getHist separately
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)  # Implement DySyn_distance separately

        best_alpha = TernarySearch(0, 1, f, 1e-2)  # Implement TernarySearch separately
        results.append(best_alpha)
        vDistAll.append(f(best_alpha))

    median_result = np.median(results)
    return [np.array([round(median_result, 2), 1 - round(median_result, 2)]), min(vDistAll)]

def PNTDiff(pos, neg, test, pos_prop):
    p_w = pos_prop / len(pos)
    n_w = (1 - pos_prop) / len(neg)
    t_w = -1 / len(test)

    p = np.column_stack((pos, np.full(len(pos), p_w)))
    n = np.column_stack((neg, np.full(len(neg), n_w)))
    t = np.column_stack((test, np.full(len(test), t_w)))

    v = np.vstack((p, n, t))
    v = v[v[:, 0].argsort()]

    acc = v[0, 1]
    total_cost = 0

    for i in range(1, len(v)):
        cost_mul = v[i, 0] - v[i - 1, 0]
        total_cost += abs(cost_mul * acc)
        acc += v[i, 1]

    return total_cost

def DySyn_SORD(p_score, n_score, test):
    def f(x):
        return PNTDiff(p_score, n_score, test, x)

    best_alpha = TernarySearch(0, 1, f, 1e-5)  # Implement TernarySearch separately
    vDist = f(best_alpha)

    return [np.array([round(best_alpha, 2), 1 - round(best_alpha, 2)]), vDist]

def X(ts, TprFpr):

    min_index = abs((1 - TprFpr[:, 1]) - TprFpr[:, 2])
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)

    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def MAX(ts, TprFpr):

    # Usual MAX implementation
    max_index = abs(TprFpr[:, 1] - TprFpr[:, 2])
    max_index = np.argmax(max_index)

    tpr_fpr_row = TprFpr[max_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def T50(ts, TprFpr):

    # Usual T50 implementation
    min_index = abs(TprFpr[:, 1] - 0.5)
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def MS(ts, TprFpr):
  results = []

  # Usual MS implementation
  threshold_set = np.arange(0.01, 1.00, 0.01)

  for threshold in threshold_set:
    threshold = round(threshold, 2) # 0.060000003 shenanigans
    tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
    dC = CC(ts, threshold)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    results.append(result)

    result = np.median(results)
  return np.array([result, 1 - result], dtype=float)

def MS2(ts, TprFpr):
    results = []

    # Usual MS2 implementation
    index = np.where(abs(TprFpr[:,1]-TprFpr[:,2]) > (1/4))[0].tolist()
    threshold_set = TprFpr[index,0]
    if threshold_set.shape[0] == 0:
      threshold_set = np.arange(0.01, 1.00, 0.01)

    # else usual MS
    for threshold in threshold_set:
      threshold = round(threshold, 2) # 0.060000003 shenanigans
      tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
      dC = CC(ts, threshold)  # Implement CC function separately

      result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
      result = max(0, min(result, 1))

      results.append(result)

    result = np.median(results)
    return np.array([result, 1 - result], dtype=float)

def PCC(ts):
    pos_score_mean = ts.mean()
    result = max(0, min(pos_score_mean, 1))

    return np.array([result, 1 - result], dtype=float)

def PACC(ts, pos_scores, neg_scores):
    dC = PCC(ts)

    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)

    result = (dC[0] - neg_mean) / (pos_mean - neg_mean) if (pos_mean - neg_mean) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def SMM(p_scores, n_scores, t_scores):
    mean_p_scores = np.mean(p_scores)
    mean_n_scores = np.mean(n_scores)
    mean_t_scores = np.mean(t_scores)

    alpha = (mean_t_scores - mean_n_scores) / (mean_p_scores - mean_n_scores)
    alpha = max(0, min(alpha, 1))

    return np.round([alpha, abs(1-alpha)], 2)

def HDy(p_score, n_score, test, err=1e-5):
    
    bins=np.linspace(10, 110, 11).astype(int)
    measure = 'hellinger'

    results = []
    distances = []

    for b_size in bins:
        Sty_1 = getHist(p_score, b_size)  # Implement getHist separately
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)  # Implement DyS_distance separately

        best_alpha = TernarySearch(0, 1, f, err)  # Implement TernarySearch separately
        results.append(best_alpha)
        distances.append(f(best_alpha))

    result = statistics.median(results)
    result = max(0, min(result, 1))

    return [np.array([result, 1 - result]), min(distances)]
