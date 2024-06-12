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
    # Value is consumed when passed to native code so in order re-use 'ar' we have to pass it by copy
    arCalc = ar.copy()
    start = timer()
    res = examply.runCalc(arCalc)
    end = timer()
    print("Python measured native loop took : ", end-start, " seconds")
    print("Nim measured native loop took : ", res[0], " ms")
    print("res=", res[1])

    start = timer()
    pyres = fLoop(ar)
    end = timer()
    print("Python loop took : ", end-start, " seconds")
    print("pyres=", pyres)

    print("2)")
    arr = examply.modArrayInPlace(ar)
    print(arr)
    # Note that arr is the same memory segment as ar since Nim code did not allocate
    print(np.shares_memory(ar, arr))

    print("----------------")
    print("END")

main()
## In bash, simply run :
## time nim c examply && time python3 examply.py > results.txt
## This takes about ~30 minutes due to long python loop
