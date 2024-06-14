import examply
import numpy as np
from timeit import default_timer as timer

def fLoop(ar):
    s = 0.0
    for i in range(0, len(ar)):
        el = ar[i]
        s = s + (1-el)/(1+el)
    print("res=", s)
    return s

def main():
    print("Python => main()")
    MAX_LEN = int(5*1e8)
    ar = np.random.rand(MAX_LEN)

    print("BEGIN")
    print("----------------")
    print(ar)

    print("1)")
    start = timer()
    res = examply.runCalc(ar)
    end = timer()
    print("Python measured native loop took : ", end-start, " seconds")
    print("Nim measured native loop took : ", res[0], " ms")
    print("res=", res[1])

    timePythonLoop = False
    # Toggle - CAREFUL it takes a long time since Python is slow
    if timePythonLoop:
        start = timer()
        pyres = fLoop(ar)
        end = timer()
        print("Python loop took : ", end-start, " seconds")
        print("pyres=", pyres)

    print("2) Showing in-place mod")
    print(ar[0:3])
    examply.modArray(ar)
    print(ar[0:3])

    print("3) Comparing for loops")

    start = timer()
    arr0 = examply.normalForOp(ar)
    end = timer()
    print("normalForOp: ", end-start, " seconds")

    start = timer()
    arr1 = examply.parallelForOp(ar)
    end = timer()
    print("parallelForOp: ", end-start, " seconds")

    if timePythonLoop:
        start = timer()
        arr2 = np.zeros(ar.shape)
        for i in range(0, len(ar)):
            arr2[i] = (1.0-ar[i])/(1.0+ar[i])
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
