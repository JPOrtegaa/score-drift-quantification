import multiprocessing
import pdb

def square(n):
    return n * n

if __name__ == "__main__":
    pdb.set_trace()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        numbers = list(range(10))
        results = pool.map(square, numbers)
        print(results)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]