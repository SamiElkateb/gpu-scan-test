from numba import cuda
import argparse
import numpy as np
import sys
import cpu
from timeit import default_timer as timer
project_gpu = __import__('project-gpu')
BENCH_DESTINATION="bench-arrays/"

def benchmark_gpu(runs, size, verification):
    print("Starting",sys._getframe(  ).f_code.co_name)
    result = np.zeros(runs, dtype=np.float32)
    for i in range(runs):
       filename = f"{BENCH_DESTINATION}/{size}_input.txt"
       input_array = np.genfromtxt(filename, delimiter=',', dtype=np.int32)
       start = timer()
       res_array = project_gpu.scan_gpu(input_array)
       cuda.synchronize()
       dt = timer() - start
       if verification:
           expected_output = np.genfromtxt(f"{BENCH_DESTINATION}/{size}_exclusive.txt", delimiter=',', dtype=np.int32)
           is_correct = np.array_equal(res_array, expected_output)
           if not is_correct:
             raise Exception("Error: output from ScanGPU was incorrect")
       print(" ", dt, " s")
       result[i]=dt
    print("Average :", np.average(result[1:]))

def benchmark_cpu(runs, size):
    print("Starting",sys._getframe(  ).f_code.co_name)
    result = np.zeros(runs, dtype=np.float32)
    for i in range(runs):
       filename = f"bench-arrays/{size}_input.txt"
       input_array = np.genfromtxt(filename, delimiter=',', dtype=np.int32)
       start = timer()
       cpu.scan_cpu(input_array)
       cuda.synchronize()
       dt = timer() - start
       print(" ", dt, " s")
       result[i]=dt
    print("Average :", np.average(result[1:]))

if __name__ =='__main__':
    parser = argparse.ArgumentParser(
       prog='Scan GPU Benchmarker',
       description='Benchmarker for the Scan GPU project in SI4 Polytech Nich Sophia-Antipolis')
    parser.add_argument('--size', '-s', type=str, default="100x1024", help='Number of times the SCAN GPU executes')
    parser.add_argument('--runs', '-r', type=int, default=10, help='Number of times the SCAN GPU executes')
    parser.add_argument('--no-verify', action='store_true', help='Only GPU Scan, No verification of the output')
    parser.add_argument('--no-cpu', action='store_true', help='Only GPU Scan no CPU scan for comparison')
    parser.add_argument('--fast', action='store_true', help='shorthand for --no-verify and --no-cpu')
    args = parser.parse_args()
    runs = args.runs
    size = args.size
    print(args)
    verification = not args.fast and not args.no_verify
    print("benchmark GPU")
    benchmark_gpu(runs, size, verification)
    if not args.fast and not args.no_cpu: 
        print("benchmark CPU")
        benchmark_cpu(runs, size)
