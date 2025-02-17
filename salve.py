import multiprocessing

def worker(i):
    print(f"Running for MF {i}")
    result = exec_eval_complexity([i])  # Ensure this returns a DataFrame
    
    # Append results to CSV immediately after each execution
    result.to_csv("results.csv", mode="a", header=False, index=False)

    return result

if __name__ == "__main__":
    m_Tr = np.arange(0.05, 1.00, 0.1)
    m_Tr = np.round(m_Tr, 2)

    print("############ - It will take a couple of minutes! - ############")

    # Write CSV headers before multiprocessing starts
    pd.DataFrame(columns=["MFtr", "MFte", "R_1", "P_1", "MAE", "Distance", "Value.dist", "Qnt"]).to_csv("results.csv", index=False)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(worker, m_Tr)  # Runs worker in parallel

    print("Processing complete. Results saved in 'results.csv'.")