import numpy as np
import pandas as pd

import os
import multiprocessing

from tqdm import tqdm

from methods.quadapt import ACCSyn, PACCSyn, XSyn, MAXSyn, T50Syn, MSSyn, MS2Syn, SMMSyn, HDySyn
from methods.quantifiers import CC, ACC, PCC, PACC, X, MAX, T50, MS, MS2, DyS, HDy, SMM, DySyn
from methods.quantifiers_utils import getTPRandFPRbyThreshold, MoSS

def apply_qntMethod(qntMethod, p_score, n_score, test, TprFpr=None, thr=None, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)):
    import mlquantify  # Ensure the mlquantify package is available
    
    if qntMethod == "CC":
        return CC(test=test, thr=thr)

    if qntMethod == "ACCSyn":
        return ACCSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "XSyn":
        return XSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MAXSyn":
        return MAXSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "T50Syn":
        return T50Syn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MSSyn":
        return MSSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MS2Syn":
        return MS2Syn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "SMMSyn":
        return SMMSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)
    
    if qntMethod == "PACCSyn":
        return PACCSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "ACC":
        return ACC(test=test, TprFpr=TprFpr, thr=thr)

    if qntMethod == "T50":
        return T50(ts=test, TprFpr=TprFpr)

    if qntMethod == "X":
        return X(ts=test, TprFpr=TprFpr)

    if qntMethod == "MAX":
        return MAX(ts=test, TprFpr=TprFpr)

    if qntMethod == "PCC":
        return PCC(ts=test)

    if qntMethod == "PACC":
        return PACC(ts=test, TprFpr=TprFpr, thr=thr)

    if qntMethod == "DySyn":
        return DySyn(ts=test, measure=measure, MF=MF_dysyn)

    if qntMethod == "HDy":
        return HDy(p_score=p_score, n_score=n_score, test=test)
    
    if qntMethod == "HDySyn":
        return HDySyn(ts=test, MF=MF_dysyn)

    if qntMethod == "DyS":
        return DyS(p_score=p_score, n_score=n_score, test=test, measure=measure)

    if qntMethod == "SORD":
        return mlquantify.SORD(p_score=p_score, n_score=n_score, test=test)

    if qntMethod == "MS":
        return MS(ts=test, TprFpr=TprFpr)

    if qntMethod == "MS2":
        return MS2(ts=test, TprFpr=TprFpr)

    if qntMethod == "SMM":
        return SMM(p_scores=p_score, n_scores=n_score, t_scores=test)

    print("ERROR - Quantification method was not applied!")
    return None

def exec_eval_complexity_single(mi, MFtr, MF_dysyn):

    # print(MFtr[mi])
    # position = (mi+1) % 30

    vdist = {"TS": "topsoe", "JD": "jensen_difference", "PS": "prob_symm", "ORD": "ord", "SORD": "sord", "TN": "taneja", "HD": "hellinger"}
    var_perc = np.arange(0, 1.01, 0.01)
    var_size = [100]
    n_tests = 10
    MF = np.arange(0.05, 1.0, 0.05)
    MF = np.round(MF, 2)
    # qnt = ['CC', 'ACC', 'ACCSyn-TS', 'ACCSyn-SORD', 'T50', 'T50Syn-TS', 'T50Syn-SORD', 'PCC', 'PACC', 'PACCSyn-TS', 'PACCSyn-SORD', 'X', 'XSyn-TS', 'XSyn-SORD', 'MAX', 'MAXSyn-TS', 'MAXSyn-SORD', 'MS', 'MSSyn-TS', 'MSSyn-SORD', 'MS2', 'MS2Syn-TS', 'MS2Syn-SORD', 'DyS-TS', 'DySyn-TS', 'DySyn-SORD', 'SMM', 'SMMSyn-TS', 'SMMSyn-SORD']
    qnt = ['CC', 'ACC', 'ACCSyn-TS', 'T50', 'T50Syn-TS', 'PCC', 'PACC', 'PACCSyn-TS', 'X', 'XSyn-TS', 'MAX', 'MAXSyn-TS', 'MS', 'MSSyn-TS', 'MS2', 'MS2Syn-TS', 'DyS-TS', 'DySyn-TS', 'SMM', 'SMMSyn-TS', 'HDy', 'HDySyn']
    results = []

    scores = MoSS(2000, 0.5, MFtr[mi])
    TprFpr = np.array(getTPRandFPRbyThreshold(scores)).astype(float)

    # description = f"Pos_prop: {MFtr[mi]}"

    for k in range(len(var_size)):
        # for i in tqdm(range(len(var_perc)), desc=description, leave=True, position=position):
        for i in range(len(var_perc)):
            for j in range(n_tests):
                for ti in range(len(MF)):
                    for qi in qnt:
                        test_set = MoSS(var_size[k], var_perc[i], MF[ti])
                        freq_REAL = pd.Series(test_set[:, 2]).value_counts(normalize=True).reindex([1, 2], fill_value=0)
                        qntMethod = qi.split("-")[0] if "-" in qi else qi

                        if qntMethod != "HDy-LP":
                            try:
                                nk = int(qntMethod.split("_")[0])
                            except ValueError:
                                nk = 1
                            qntMethod = "DySyn" if nk != 1 else qntMethod

                        measure = None
                        if len(qi.split("-")) > 1:
                            measure = vdist.get(qi.split("-")[1])

                        qnt_re = apply_qntMethod(
                            qntMethod=qntMethod,
                            p_score=scores[scores[:, 2] == 1, 0],
                            n_score=scores[scores[:, 2] == 2, 0],
                            test=test_set[:, 0],
                            TprFpr=TprFpr,
                            thr=0.5,
                            measure=measure,
                            MF_dysyn=MF_dysyn,
                        )

                        if qntMethod in ["DySyn", "HDy", "HDySyn"]:
                            freq_PRE = np.round(qnt_re[0][0], 3)
                        else:
                            freq_PRE = np.round(qnt_re[0], 3)

                        results.append([
                            MFtr[mi],
                            MF[ti],
                            freq_REAL.get(1, 0),
                            freq_PRE,
                            np.round(abs(freq_REAL.get(1, 0) - freq_PRE), 2),
                            measure,
                            # qnt_re[1],
                            qi,
                        ])
    return results

def worker(mi_MFtr_MFdysyn):
    mi, MFtr, MF_dysyn = mi_MFtr_MFdysyn
    results = exec_eval_complexity_single(mi, MFtr, MF_dysyn)
    
    # df = pd.DataFrame(results, columns=["MFtr", "MFte", "R_1", "P_1", "AE", "Distance", "Qnt"])
    # df.to_csv("results_syn.csv", mode="a", header=False, index=False)

    return results  # No need to return anything it was NONE before!!!!

def exec_eval_complexity_parallel(MFtr, dysyn_range):

    for MF_dysyn in tqdm(dysyn_range, desc="MF_dysyn combinations", leave=True, position=0):
        # Prepare arguments for each worker
        start, finish, step = MF_dysyn["start"], MF_dysyn["finish"], MF_dysyn["step"]
        
        if start == finish:
            # Make sure MF_dysyn is a 1-D array even for a single value
            MF_dysyn = np.array([np.round(start, 2)], dtype=float)
        else:
            MF_dysyn = np.round(np.arange(start, finish, step), 2)
        
        tasks = [(mi, MFtr, MF_dysyn) for mi in range(len(MFtr))]

        # Write CSV header before multiprocessing starts
        file_name = "./ablation/results_syn_" + str(start) + "_" + str(finish) + "_" + str(step) + ".csv"
        if not os.path.exists(file_name):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            pd.DataFrame(columns=["MFtr", "MFte", "R_1", "P_1", "AE", "Distance", "Qnt"]).to_csv(file_name, index=False)


        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(pool.map(worker, tasks))

        # Flatten results and convert to DataFrame
        flat_results = [item for sublist in results for item in sublist]
        df = pd.DataFrame(flat_results, columns=["MFtr", "MFte", "R_1", "P_1", "AE", "Distance", "Qnt"])
        df.to_csv(file_name, mode="a", header=False, index=False)

        print("All workers finished.")

def generate_combinations(step_array):
    # Store only start, finish, and step values for each combination
    dysyn_range = []
    start = 0.0
    finish = 1.0

    while start < finish:
        for step in step_array:
            dysyn_range.append({"start": np.round(start, 2), "finish": np.round(finish, 2), "step": np.round(step, 2)})
        start += 0.1
        finish -= 0.1

    return dysyn_range

if __name__ == "__main__":
    m_Tr = np.arange(0.05, 1.0, 0.05)
    m_Tr = np.round(m_Tr, 2)
    # MF_dysyn = np.arange(0.1, 1.0, 0.2)

    step_array = np.round(np.arange(0,0.55,0.05), 2)
    step_array[0] = np.round(0.01, 2)

    dysyn_range = generate_combinations(step_array)

    dysyn_server = dysyn_range[:18]
    dysyn_home = dysyn_range[18:34]
    dysyn_luiz = dysyn_range[34:50]
    dysyn_rafael = dysyn_range[50:]

    # # Remove items in dysyn_home before start=0.2, finish=0.8, step=0.25
    # for idx, item in enumerate(dysyn_home):
    #     if item["start"] == 0.2 and item["finish"] == 0.8 and item["step"] == 0.25:
    #         dysyn_home = dysyn_home[idx:]
    #         break

    for dysyn in [dysyn_rafael]:
        exec_eval_complexity_parallel(m_Tr, [dysyn[5]])

    print("Experiment complete!")