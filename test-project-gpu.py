import unittest
import subprocess
import numpy as np
import sys
import cpu
from pathlib import Path
import argparse
from env import MEMCHECK, PROJECT_FILE, TEST_FILE_NAME, BENCH_DESTINATION, LARGE_ARRAY_SIZES
number_generator = __import__('number-generator')

COURSE_ARRAY_1 = [2, 3, 4, 6]
COURSE_EXPECTED_1 = [0, 2, 5, 9]
COURSE_EXPECTED_1_INCLUSIVE = [2, 5, 9, 15]

COURSE_ARRAY_2 = [1, 3, 4, 12, 2, 7, 0, 4]
COURSE_EXPECTED_2 = [0, 1, 4, 8, 20, 22, 29, 29]
COURSE_EXPECTED_2_INCLUSIVE = [1, 4, 8, 20, 22, 29, 29, 33]
COURSE_EXPECTED_2_INDEPENDENT = [0, 1, 0, 4, 0, 2, 0, 0]

COURSE_ARRAY_3 = [2, 9, 15, 13, 10, 20, 2, 3]
COURSE_EXPECTED_3 = [0, 2, 11, 26, 39, 49, 69, 71]
COURSE_EXPECTED_3_INCLUSIVE = [2, 11, 26, 39, 49, 69, 71, 74]
COURSE_EXPECTED_3_INDEPENDENT = [0, 2, 0, 15, 0, 10, 0, 2]

TEST_LARGE_ARRAYS = True
TEST_MEMORY = True
NUMBER_RANDOM_TESTS = 1
IS_DEBUG = False


def array_to_string(arr):
    return (np.array2string(arr, separator=",", threshold=arr.shape[0]).strip('[]').replace('\n', '').replace(' ',''))


def array_to_file(arr, filename=TEST_FILE_NAME):
    string_to_file(array_to_string(arr), filename)


def string_to_file(string, filename=TEST_FILE_NAME):
    with open(filename, "w") as file:
        file.write(string)


class TestScanExamples(unittest.TestCase):
    def base_test(self, input_array, output_array, thread_block_size=None, independent=None, inclusive=None, name=""):
        input_array = np.array(input_array)
        array_to_file(input_array)
        expected_array = np.array(output_array)
        thread_block_arg = "" if thread_block_size is None else f"--tb {thread_block_size}"
        independent_arg = "--independent" if independent is True else ""
        inclusive_arg = "--inclusive" if inclusive is True else ""
        standard_cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        expected_output = array_to_string(expected_array)
        standardcmd_stdout = standard_cmd.stdout.decode()
        if IS_DEBUG:
            standardcmd_stderr = standard_cmd.stderr.decode()
            print(f"Debug: {standardcmd_stderr}")
        self.assertEqual(standardcmd_stdout, expected_output, msg=f"{name} Failed")
        print(f"{name} Succeeded")
        if not TEST_MEMORY:
            return
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0', f"{name} Memcheck Failed")
        print(f"{name} Memcheck Succeeded")

    def test_with_array_creation(self, array_size=32, thread_block_size=None, independent=None, inclusive=None, name=""):
        input_array = number_generator.create_test_file(arbitrary_size=array_size, file_name=TEST_FILE_NAME)
        expected_output = cpu.get_expected_output_inclusive(input_array) if inclusive is True else cpu.get_expected_output(input_array)
        thread_block_arg = "" if thread_block_size is None else f"--tb {thread_block_size}"
        independent_arg = "--independent" if independent is True else ""
        inclusive_arg = "--inclusive" if inclusive is True else ""
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        stdout = cmd.stdout.decode()
        if IS_DEBUG:
            cmd_stderr = cmd.stderr.decode()
            print(f"Debug: {cmd_stderr}")
        self.assertEqual(stdout, expected_output, f"{name} Failed")
        print(f"{name} Succeeded")
        if not TEST_MEMORY:
            return
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0', f"{name} Memcheck Failed")
        print(f"{name} Memcheck Succeeded")

    def benchmark_test(self, size="100x1024", thread_block_size=None, inclusive=None, name=""):
        thread_block_arg = "" if thread_block_size is None else f"--tb {thread_block_size}"
        inclusive_arg = "--inclusive" if inclusive is True else ""
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {BENCH_DESTINATION}{size}_input.txt {thread_block_arg} {inclusive_arg}", capture_output=True, shell=True)
        expected_endfile = "inclusive" if inclusive is True else "exclusive"
        expected_output = Path(f"{BENCH_DESTINATION}{size}_{expected_endfile}.txt").read_text()
        stdout = cmd.stdout.decode()
        if IS_DEBUG:
            cmd_stderr = cmd.stderr.decode()
            print(f"Debug: {cmd_stderr}")
        self.assertEqual(stdout, expected_output, f"{name} Failed")
        print(f"{name} Succeeded")
        if not TEST_MEMORY:
            return
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {BENCH_DESTINATION}{size}_input.txt {thread_block_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0', f"{name} Memcheck Failed")
        print(f"{name} Memcheck Succeeded")

    def test_exclusive_scan_course_1(self):
        name = "First example of the course"
        print(f"\n{name}")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, name=name)

    def test_exclusive_scan_course_2(self):
        name = "Second example of the course"
        print(f"\n{name}")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, name=name)

    def test_exclusive_scan_course_3(self):
        name = "Third example of the course"
        print(f"\n{name}")
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, name=name)

    def test_exclusive_scan_course_larger_thread_block(self):
        print("\nAll examples of the course with thread block larger than array size")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, thread_block_size=8, name="First course example")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, thread_block_size=16, name="Second course example")
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, thread_block_size=16, name="Third course example")

    def test_exclusive_scan_course_smaller_thread_block(self):
        print("\nAll examples of the course with thread block smaller than array size")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, thread_block_size=2, name="First course example")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, thread_block_size=2, name="Second course example")
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, thread_block_size=2, name="Third course example")

    def test_exclusive_scan_course_array_not_pow2(self):
        print("\nAll examples of the course modified with array size not power of 2")
        self.base_test(input_array=COURSE_ARRAY_1[:-1], output_array=COURSE_EXPECTED_1[:-1], name="First course example")
        self.base_test(input_array=COURSE_ARRAY_2[:-1], output_array=COURSE_EXPECTED_2[:-1], name="Second course Example")
        self.base_test(input_array=COURSE_ARRAY_3[:-1], output_array=COURSE_EXPECTED_3[:-1], name="Third course example")

    def test_inclusive_scan_course(self):
        print("\nAll examples of the course as inclusive scan")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1_INCLUSIVE, inclusive=True, name="First course example")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2_INCLUSIVE, inclusive=True, name="Second course example")
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3_INCLUSIVE, inclusive=True, name="Third course example")

    def test_independent_scan_course(self):
        print("\nAll examples of the course as inclusive scan")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2_INDEPENDENT, thread_block_size=2, independent=True, name="Second course example")
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3_INDEPENDENT, thread_block_size=2, independent=True, name="Third course example")

    def test_exclusive_scan_single_equal_thread_block(self):
        print("\nExclusive scan: Array of size 2^m with equal single thread block (32, 32)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 32
            self.test_with_array_creation(array_size=array_size, thread_block_size=array_size, name=f"Test n°{i}")

    def test_exclusive_scan_single_larger_thread_block(self):
        print("\nExclusive scan: Array of size 2^m with a too large thread block (32, 64)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 32
            block_size = 64
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

    def test_exclusive_scan_single_thread_block_not_pow2(self):
        print("\nExclusive scan: Array of size !2^m with a single thread block (30, 32)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 30
            block_size = 32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

    def test_exclusive_scan_arbitrary_pow2(self):
        print("\nArbitrary scans of: Array of size 2^m with multiple thread block (64, 8)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 64
            block_size = 16
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

    def test_exclusive_scan_arbitrary_not_pow2(self):
        print("\nArbitrary scans of: Array of size 2^m with multiple thread block (1023, 8)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 1023
            block_size = 8
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

    def test_exclusive_scan_arbitrary_pow2_large_array(self):
        print("\nExclusive scan: Very large array (1025, 1024)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 1025
            block_size = 1024
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

    def test_inclusive_scan(self):
        print("\nInclusive scan (128, 32)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 128
            block_size = 32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, inclusive=True, name=f"Test n°{i}")

    def test_exclusive_scan_large_arrays(self):
        if TEST_LARGE_ARRAYS is False:
            self.skipTest("Skipping large arrays inclusive")
        for _, size in enumerate(LARGE_ARRAY_SIZES):
            print(f"Exclusive scan: large arrays: {size}")
            self.benchmark_test(size=size)

    def test_inclusive_scan_large_arrays(self):
        if TEST_LARGE_ARRAYS is False:
            self.skipTest("Skipping large arrays inclusive")
        for _, size in enumerate(LARGE_ARRAY_SIZES):
            print(f"Inclusive scan: large arrays: {size}")
            self.benchmark_test(size=size, inclusive=True)

    def test_exclusive_block_size_not_pow_2(self):
        print("\nExclusive scan: block size not pow 2 (5, 3)")
        for i in range(1, NUMBER_RANDOM_TESTS + 1):
            array_size = 5
            block_size = 3
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name=f"Test n°{i}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Scan GPU Tester',
        description='Tests for the Scan GPU project in SI4 Polytech Nich Sophia-Antipolis')
    parser.add_argument('--random-tests', type=int, default=1)
    parser.add_argument('--no-large-arrays', action='store_true')
    parser.add_argument('--no-mem-check', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    TEST_LARGE_ARRAYS = not args.no_large_arrays
    TEST_MEMORY = not args.no_mem_check
    NUMBER_RANDOM_TESTS = args.random_tests
    IS_DEBUG = args.debug

    print("Starting tests: ")
    print("With memory checks") if TEST_MEMORY else print("Without memory checks")
    print("With large arrays") if LARGE_ARRAY_SIZES else print("Without large arrays")
    print()
        
    unittest.main(argv=[sys.argv[0], "--failfast"])

