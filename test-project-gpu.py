import unittest
import number_generator
import subprocess
import numpy as np
import cpu
from pathlib import Path

NUMBER_RANDOM_TESTS=1
MEMCHECK="/usr/local/cuda/bin/compute-sanitizer"
PROJECT_FILE="project-gpu.py"
TEST_FILE_NAME="testfile.txt"
RES_FILENAME="resfile.txt"
BENCH_DESTINATION="bench-arrays/"

COURSE_ARRAY_1=[2, 3, 4, 6]
COURSE_EXPECTED_1=[0, 2, 5, 9]
COURSE_EXPECTED_1_INCLUSIVE=[2, 5, 9, 15]

COURSE_ARRAY_2=[1, 3, 4, 12, 2, 7, 0, 4]
COURSE_EXPECTED_2=[0, 1, 4, 8, 20, 22, 29, 29]
COURSE_EXPECTED_2_INCLUSIVE=[1, 4, 8, 20, 22, 29, 29, 33]
COURSE_EXPECTED_2_INDEPENDENT=[0, 1, 0, 4, 0, 2, 0, 0]

COURSE_ARRAY_3=[2, 9, 15, 13, 10, 20, 2, 3]
COURSE_EXPECTED_3=[0, 2, 11, 26, 39, 49, 69, 71]
COURSE_EXPECTED_3_INCLUSIVE=[2, 11, 26, 39, 49, 69, 71, 74]
COURSE_EXPECTED_3_INDEPENDENT=[0, 2, 0, 15, 0, 10, 0, 2]

LARGE_ARRAY_SIZES=["100x1024", "200x1024", "300x1024", "400x1024", "500x1024", "600x1024", "700x1024", "800x1024", "900x1024"]

def array_to_string(arr):
    return (np.array2string(arr, separator=",", threshold=arr.shape[0]).strip('[]').replace('\n', '').replace(' ',''))

def array_to_file(arr, filename=TEST_FILE_NAME):
    string_to_file(array_to_string(arr), filename)

def string_to_file(string, filename=TEST_FILE_NAME):
  with open(filename, "w") as file:
     file.write(string)

class TestScanExamples(unittest.TestCase):
    def base_test(self, input_array, output_array, thread_block_size=None, independent=None, inclusive=None, name=""):
        if name != "":
            print(name)
        input_array = np.array(input_array)
        array_to_file(input_array)
        expected_array = np.array(output_array)
        thread_block_arg = "" if thread_block_size == None else f"--tb {thread_block_size}"
        independent_arg = f"--independent" if independent == True else ""
        inclusive_arg = f"--inclusive" if inclusive == True else ""
        standard_cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        expected_output = array_to_string(expected_array)
        standardcmd_stdout = standard_cmd.stdout.decode()
        self.assertEqual(standardcmd_stdout, expected_output)
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0')
    
    def test_with_array_creation(self, array_size=32, thread_block_size=None, independent=None, inclusive=None, name=""):
        if name != "":
            print(name)
        input_array = number_generator.create_test_file(arbitrary_size=array_size, file_name=TEST_FILE_NAME)
        expected_output = cpu.get_expected_output_inclusive(input_array) if inclusive == True else cpu.get_expected_output(input_array)
        thread_block_arg = "" if thread_block_size == None else f"--tb {thread_block_size}"
        independent_arg = f"--independent" if independent == True else ""
        inclusive_arg = f"--inclusive" if inclusive == True else ""
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        stdout = cmd.stdout.decode()
        self.assertEqual(stdout, expected_output)
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0')

    def create_bench_data(self, array_size=32, thread_block_size=None, independent=None, inclusive=None, name=""):
        if name != "":
            print(name)
        input_array = number_generator.create_test_file(arbitrary_size=array_size, file_name=TEST_FILE_NAME)
        expected_output = cpu.get_expected_output_inclusive(input_array) if inclusive == True else cpu.get_expected_output(input_array)
        thread_block_arg = "" if thread_block_size == None else f"--tb {thread_block_size}"
        independent_arg = f"--independent" if independent == True else ""
        inclusive_arg = f"--inclusive" if inclusive == True else ""
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        stdout = cmd.stdout.decode()
        self.assertEqual(stdout, expected_output)
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0')
        start_filename=f"{BENCH_DESTINATION}{int(array_size/1024)}x1024"
        input_filename=f"{start_filename}_input.txt"
        expected_filename=f"{start_filename}_exclusive.txt"
        array_to_file(input_array, input_filename)
        string_to_file(expected_output, expected_filename)

    def benchmark_test(self, size="100x1024", thread_block_size=None, inclusive=None, name=""):
        if name != "":
            print(name)
        thread_block_arg = "" if thread_block_size == None else f"--tb {thread_block_size}"
        inclusive_arg = f"--inclusive" if inclusive == True else ""
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {BENCH_DESTINATION}{size}_input.txt {thread_block_arg} {inclusive_arg}", capture_output=True, shell=True)
        expected_endfile = "inclusive" if inclusive == True else "exclusive"
        expected_output = Path(f"{BENCH_DESTINATION}{size}_{expected_endfile}.txt").read_text()
        stdout = cmd.stdout.decode()
        self.assertEqual(stdout, expected_output)
        memcheck_cmd = subprocess.run(f"{MEMCHECK} python3 {PROJECT_FILE} {BENCH_DESTINATION}{size}_input.txt {thread_block_arg} {inclusive_arg}", capture_output=True, shell=True)
        memcheck_cmd = memcheck_cmd.stdout.decode()
        self.assertRegex(memcheck_cmd, r'========= ERROR SUMMARY: 0')
        
        
    def test_exclusive_scan_course_1(self):
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, name="First example of the course")

    def test_exclusive_scan_course_2(self):
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, name="Second example of the course")

    def test_exclusive_scan_course_3(self):
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, name="Third example of the course")

    def test_exclusive_scan_course_larger_thread_block(self):
        print("All examples of the course with thread block larger than array size")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, thread_block_size=8)
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, thread_block_size=16)
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, thread_block_size=16)

    def test_exclusive_scan_course_smaller_thread_block(self):
        print("All examples of the course with thread block smaller than array size")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1, thread_block_size=2)
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2, thread_block_size=2)
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3, thread_block_size=2)

    def test_exclusive_scan_course_array_not_pow2(self):
        print("All examples of the course modified with array size not power of 2")
        self.base_test(input_array=COURSE_ARRAY_1[:-1], output_array=COURSE_EXPECTED_1[:-1])
        self.base_test(input_array=COURSE_ARRAY_2[:-1], output_array=COURSE_EXPECTED_2[:-1])
        self.base_test(input_array=COURSE_ARRAY_3[:-1], output_array=COURSE_EXPECTED_3[:-1])

    def test_inclusive_scan_course(self):
        print("All examples of the course as inclusive scan")
        self.base_test(input_array=COURSE_ARRAY_1, output_array=COURSE_EXPECTED_1_INCLUSIVE, inclusive=True)
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2_INCLUSIVE, inclusive=True)
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3_INCLUSIVE, inclusive=True)

    def test_independent_scan_course(self):
        print("All examples of the course as inclusive scan")
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2_INDEPENDENT, thread_block_size=2, independent=True)
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3_INDEPENDENT, thread_block_size=2, independent=True)


    def test_exclusive_scan_single_equal_thread_block(self):
        print("Exclusive scan: Array of size 2^m with equal single thread block (32, 32)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=32
            self.test_with_array_creation(array_size=array_size, thread_block_size=array_size, name="")
        

    def test_exclusive_scan_single_larger_thread_block(self):
        print("Exclusive scan: Array of size 2^m with a too large thread block (32, 64)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=32
            block_size=64
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_single_thread_block_not_pow2(self):
        print("Exclusive scan: Array of size !2^m with a single thread block (30, 32)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=30
            block_size=32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_arbitrary_pow2(self):
        print("Arbitrary scans of: Array of size 2^m with multiple thread block (64, 8)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=64
            block_size=16
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_arbitrary_not_pow2(self):
        print("Arbitrary scans of: Array of size 2^m with multiple thread block (1023, 8)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=1023
            block_size=8
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_arbitrary_pow2_large_array(self):
        print("Exclusive scan: Very large array (1025, 1024)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=1025
            block_size=1024
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_inclusive_scan(self):
        print("Inclusive scan (128, 32)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=128
            block_size=32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, inclusive=True, name="")

    def test_exclusive_scan_large_arrays(self):
        for i, size in enumerate(LARGE_ARRAY_SIZES):
            print(f"Exclusive scan: large arrays: {size}")
            self.benchmark_test(size=size)

    def test_inclusive_scan_large_arrays(self):
        for i, size in enumerate(LARGE_ARRAY_SIZES):
            print(f"Exclusive scan: large arrays: {size}")
            self.benchmark_test(size=size, inclusive=True)

if __name__ == '__main__':
    unittest.main()
