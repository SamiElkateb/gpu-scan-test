from numba import cuda
import numpy as np
import math

def descente_cpu(arr: np.ndarray):
    n = arr.size
    m = (int)(math.log2(n))
    arr[n-1] = 0
    for d in range(m-1, -1, -1):
        for k in range(0, n-1, 2**(d+1)):
            tmp = arr[k+2**d-1]
            arr[k+2**d-1] = arr[k+2**(d+1)-1]
            arr[k+(2**(d+1)-1)]+=tmp
    return arr

def montee_cpu(arr: np.ndarray):
    n = arr.size
    for d in range(0, (int)(math.log2(n))):
        for k in range(0, n-1, 2**(d+1)):
            arr[k+2**(d+1)-1]+= arr[k+2**d-1]
    return arr

def scan_cpu(array: np.ndarray):
    orig_size = array.shape[0]
    n = 2**math.ceil(math.log2(array.shape[0]))
    if n != array.shape[0]:
        array = np.pad(array, (0, n - array.shape[0]), 'constant', constant_values=0)

    array = montee_cpu(array)
    array = descente_cpu(array)
    array.resize(orig_size)
    return array

def prefix_sum_loop(arr: np.ndarray):
    acc = 0
    for i in range(0, arr.size):
        tmp = arr[i]
        arr[i] = acc
        acc += tmp
    return arr

def get_expected_output(array: np.ndarray):
    res = scan_cpu(array)
    return (np.array2string(res, separator=",", threshold=res.shape[0]).strip('[]').replace('\n', '').replace(' ',''))

def get_expected_output_inclusive(array: np.ndarray):
    arr_copy = np.copy(array)
    res_exclusive_scan = scan_cpu(array)
    res = np.add(arr_copy, res_exclusive_scan)  
    return (np.array2string(res, separator=",", threshold=res.shape[0]).strip('[]').replace('\n', '').replace(' ',''))


def main():
    arr = np.array([2, 3, 4, 6], dtype=np.int32)
    # res = prefix_sum_loop(arr)
    res = scan_cpu(arr)
    print(res);
   

if __name__ == '__main__':
    main()