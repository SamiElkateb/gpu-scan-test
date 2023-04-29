import sys
import numpy


def usage():
    print("Usage: " + sys.argv[0] + " <power of 2>")


def arbitrary_size_array(size, min=-100, max=100):
    return numpy.random.randint(min, max, size, dtype=numpy.int32)


def pow_2_array(power_of_two, min=-100, max=100):
    return numpy.random.randint(min, max, 2**power_of_two, dtype=numpy.int32)


def create_test_file(pow_of_two=None, file_name='testfile.txt', min=-100, max=100, arbitrary_size=None):
    test_array = []
    test_array_string = ""
    if pow_of_two is not None and arbitrary_size is not None:
        raise Exception("Test file size cannot be power of to and arbitrary size simultaneously")
    if pow_of_two is None and arbitrary_size is None:
        raise Exception("Provide a pow_of_two or an arbitrary_size")
    if pow_of_two is not None:
        test_array = pow_2_array(pow_of_two, min, max)
        test_array_string = (numpy.array2string(test_array, separator=",", threshold=test_array.shape[0]).strip('[]').replace('\n', '').replace(' ',''))
    if arbitrary_size is not None:
        test_array = arbitrary_size_array(arbitrary_size, min, max)
        test_array_string = (numpy.array2string(test_array, separator=",", threshold=test_array.shape[0]).strip('[]').replace('\n', '').replace(' ',''))
    with open(file_name, "w") as test_file:
        test_file.write(test_array_string)
    return test_array


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        sys.exit(2)
    else:
        create_test_file(int(sys.argv[1]))

