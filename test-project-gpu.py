import unittest
import number_generator
import subprocess
import numpy as np
import cpu
from random import randint

NUMBER_RANDOM_TESTS=1

PROJECT_FILE="project-gpu.py"
TEST_FILE_NAME="testfile.txt"
RES_FILENAME="resfile.txt"

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

def array_to_string(arr):
    return (np.array2string(arr, separator=",", threshold=arr.shape[0]).strip('[]').replace('\n', '').replace(' ',''))

def array_to_file(arr):
  with open(TEST_FILE_NAME, "w") as test_file:
     test_file.write(array_to_string(arr))

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
        cmd = subprocess.run(f"python3 {PROJECT_FILE} {TEST_FILE_NAME} {thread_block_arg} {independent_arg} {inclusive_arg}", capture_output=True, shell=True)
        expected_output = array_to_string(expected_array)
        stdout = cmd.stdout.decode()
        self.assertEqual(stdout, expected_output)
    
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
        self.base_test(input_array=COURSE_ARRAY_2, output_array=COURSE_EXPECTED_2_INDEPENDENT, independent=True)
        self.base_test(input_array=COURSE_ARRAY_3, output_array=COURSE_EXPECTED_3_INDEPENDENT, independent=True)


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

    def test_exclusive_scan_arbitrary_pow_not_pow2(self):
        print("Arbitrary scans of: Array of size 2^m with multiple thread block (64, 8)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=64
            block_size=8
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_arbitrary_pow_not_pow2(self):
        print("Arbitrary scans of: Array of size 2^m with multiple thread block (64, 8)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=1023
            block_size=8
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_exclusive_scan_arbitrary_pow2_very_large_array(self):
        print("Exclusive scan: Very large array (2^25, 32)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=1025
            block_size=32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, name="")

    def test_inclusive_scan(self):
        print("Inclusive scan (128, 32)")
        for i in range(NUMBER_RANDOM_TESTS):
            array_size=128
            block_size=32
            self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, inclusive=True, name="")

    # def test_inclusive_scan_pow2_very_large_array(self):
    #     print("Inclusive scan: Very large array")
    #     for i in range(NUMBER_RANDOM_TESTS):
    #         array_size=2**25
    #         block_size=8
    #         self.test_with_array_creation(array_size=array_size, thread_block_size=block_size, inclusive=True, name="")


if __name__ == '__main__':
    unittest.main()