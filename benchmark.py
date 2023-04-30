from numba import cuda
import argparse
import numpy as np
import sys
import cpu
from timeit import default_timer as timer
from env import PROJECT_FILE, BENCH_DESTINATION

project_gpu = __import__(PROJECT_FILE.replace('.py', ''))


def local_benchmark_gpu(runs, size, inclusive=False):
    print("Starting", sys._getframe().f_code.co_name)
    result = np.zeros(runs, dtype=np.float32)
    for i in range(runs):
        array = np.array([1 for _ in range(size)], dtype=np.int32)
        start = timer()
        print(project_gpu.scan_gpu(array, inclusive=inclusive))
        cuda.synchronize()
        dt = timer() - start
        print(" ", dt, " s")
        result[i] = dt
    print("Average :", np.average(result[1:]))


def benchmark_gpu(runs, size, warmups, verification, inclusive=False):
    print("Starting", sys._getframe().f_code.co_name)
    result = np.zeros(runs, dtype=np.float32)
    filename = f"{BENCH_DESTINATION}/{size}_input.txt"
    base_array = np.genfromtxt(filename, delimiter=',', dtype=np.int32)
    for i in range(warmups):
        input_array = np.copy(base_array)
        project_gpu.scan_gpu(input_array, inclusive=inclusive)
        cuda.synchronize()

    if verification:
        endfile = "inclusive" if inclusive else "exclusive"
        expected_output = np.genfromtxt(f"{BENCH_DESTINATION}/{size}_{endfile}.txt", delimiter=',', dtype=np.int32)

    for i in range(runs):
        start = timer()
        input_array = np.copy(base_array)
        res_array = project_gpu.scan_gpu(input_array, inclusive=inclusive)
        cuda.synchronize()
        dt = timer() - start
        if verification:
            is_correct = np.array_equal(res_array, expected_output)
            if not is_correct:
                raise Exception("Error: output from ScanGPU was incorrect")
        print(" ", dt, " s")
        result[i] = dt
    print("Average :", np.average(result[1:]))


def benchmark_cpu(runs, size):
    print("Starting", sys._getframe().f_code.co_name)
    result = np.zeros(runs, dtype=np.float32)
    for i in range(runs):
        filename = f"bench-arrays/{size}_input.txt"
        input_array = np.genfromtxt(filename, delimiter=',', dtype=np.int32)
        start = timer()
        cpu.scan_cpu(input_array)
        cuda.synchronize()
        dt = timer() - start
        print(" ", dt, " s")
        result[i] = dt
    print("Average :", np.average(result[1:]))


# if __name__ == '__main__':
#     local_benchmark_gpu(2, 100_000_000, inclusive=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Scan GPU Benchmarker',
        description='Benchmarker for the Scan GPU project in SI4 Polytech Nich Sophia-Antipolis')

    parser.add_argument('--size', '-s', type=str, default="100x1024", help='Number of times the SCAN GPU executes')
    parser.add_argument('--runs', '-r', type=int, default=10, help='Number of times the SCAN GPU executes')
    parser.add_argument('--warmups', type=int, default=5, help='Number of times the SCAN GPU executes without adding the data to the average')
    parser.add_argument('--inclusive', action='store_true')
    parser.add_argument('--no-verify', action='store_true', help='Only GPU Scan, No verification of the output')
    parser.add_argument('--no-cpu', action='store_true', help='Only GPU Scan no CPU scan for comparison')
    parser.add_argument('--fast', action='store_true', help='shorthand for --no-verify and --no-cpu')
    args = parser.parse_args()
    runs = args.runs
    size = args.size
    warmups = args.warmups
    inclusive = args.inclusive
    verification = not args.fast and not args.no_verify
    print("benchmark GPU")
    benchmark_gpu(runs, size, warmups, verification, inclusive=inclusive)
    if not args.fast and not args.no_cpu:
        print("benchmark CPU")
        benchmark_cpu(runs, size)

