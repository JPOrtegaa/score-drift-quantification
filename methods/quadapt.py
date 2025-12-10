import numpy as np

from .Quantifiers import CC, PCC, T50, DySyn, HDy, SMM
from .Quantifiers_Utils import getTPRandFPRbyThreshold, MoSS

def ACCSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)  # Implement getTPRandFPRbyThreshold

    dC = CC(ts)  # Implement CC function separately

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == 0.5, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def XSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)
    
    # Usual X implementation
    min_index = abs((1 - TprFpr[:, 1]) - TprFpr[:, 2])
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)
    # if tpr_fpr_row.size == 0:
    #   raise ValueError("Threshold value not found in TprFpr.")

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def MAXSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    # Usual MAX implementation
    max_index = abs(TprFpr[:, 1] - TprFpr[:, 2])
    max_index = np.argmax(max_index)

    tpr_fpr_row = TprFpr[max_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def T50Syn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    return T50(ts, TprFpr)  # Reuse the T50 function

def MSSyn(ts, measure, MF_dysyn):
    results = []

    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

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

def MS2Syn(ts, measure, MF_dysyn):
    results = []

    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)
    
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

def PACCSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)  # Implement getTPRandFPRbyThreshold

    dC = PCC(ts)

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == 0.5, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def SMMSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    best_scores = MoSS(1000, 0.5, rQnt[2]) # sempre NaN em 0 de distances

    best_p = best_scores[best_scores[:, 2] == 1, 0]
    best_n = best_scores[best_scores[:, 2] == 2, 0]
    result = SMM(best_p, best_n, ts)

    return result

def HDySyn(ts, MF=np.arange(0.1, 1.0, 0.2)):
    # Ensure MF is always iterable (handles scalar numpy.float64 / 0-d arrays)
    MF = np.atleast_1d(np.round(MF, 2)).astype(float) # When MF with only 0.5 value was evaluated, instead of an array of MF.

    results = []
    distances = []

    for mf in MF:
        scores = MoSS(1000, 0.5, mf)  # Implement MoSS function separately
        test_p = scores[scores[:, 2] == 1, 0]
        test_n = scores[scores[:, 2] == 2, 0]

        rQnt = HDy(test_p, test_n, ts)  # Implement DySyn_DyS separately

        distances.append(rQnt[1])
        results.append(rQnt[0][0])

    best_result = round(results[np.argmin(distances)], 2)
    return [np.array([best_result, 1 - best_result]), min(distances), MF[np.argmin(distances)]]