import examply
import numpy as np
from timeit import default_timer as timer
import multiprocessing

def fLoop(ar):
    s = 0.0
    iX = int(ar.shape[0])
    iY = int(ar.shape[1])

    for i in range(0, iX):
        for j in range(0, iY):
            el = ar[i, j]
            s = s + (1-el)/(1+el)
    print("res=", s)
    return s

def main():
    print("Python => main()")
    MAX_X = int(3*1e3)
    MAX_Y = int(4*1e4)
    MAX_LEN = int(MAX_X*MAX_Y)
    print("CPU COUNT=",  multiprocessing.cpu_count())
    print("MAX_X=", MAX_X)
    print("MAX_Y=", MAX_Y)
    print("MAX_LEN=", MAX_LEN)
    ar = np.random.rand(MAX_X, MAX_Y)

    print("BEGIN")
    print("----------------")
    print(ar)

    print("1)")
    timePythonLoop = True
    # Toggle - CAREFUL it takes a long time since Python is slow
    if timePythonLoop:
        start = timer()
        pyres = fLoop(ar)
        end = timer()
        print("Python loop took : ", end-start, " seconds")
        print("pyres=", pyres)

    start = timer()
    res = examply.runCalc(ar)
    end = timer()
    print("Python measured native loop took : ", end-start, " seconds")
    print("Nim measured native loop took : ", res[0], " ms")
    print("res=", res[1])

    print("2) Showing in-place mod")
    print(ar[0, 0:3])
    examply.modArray(ar)
    print(ar[0, 0:3])

    print("3) Comparing for loops")

    start = timer()
    arr0 = examply.normalForOp(ar)
    end = timer()
    print("normalForOp: ", end-start, " seconds")

    start = timer()
    arr01 = examply.indexedOp(ar)
    end = timer()
    print("indexedOp: ", end-start, " seconds")

    start = timer()
    arr1 = examply.parallelForOp(ar)
    end = timer()
    print("parallelForOp: ", end-start, " seconds")

    start = timer()
    arr11 = examply.parallelIndexedForOp(ar)
    end = timer()
    print("parallelIndexedForOp: ", end-start, " seconds")

    if timePythonLoop:
        start = timer()
        arr2 = np.zeros(ar.shape)
        X = int(ar.shape[0])
        Y = int(ar.shape[1])
        for i in range(0, X):
            for j in range(0, Y):
                arr2[i, j] = (1.0-ar[i, j])/(1.0+ar[i, j])
        end = timer()
        print("Native python for: ", end-start, " seconds")

    # We can check that it returns a copy
    print(np.shares_memory(ar, arr0))
    print(np.shares_memory(ar, arr1))

    # Check results are identical
    eq = np.allclose(arr0, arr1)
    if timePythonLoop:
        eq = np.allclose(arr0, arr2)
    print(eq)

    print("----------------")
    print("END")

main()
## In bash, simply run :
## time nim c examply && time python3 examply.py > results.txt
## This takes about ~30 minutes due to long python loop
