import examply
import numpy as np
from timeit import default_timer as timer

def timeMe(text="func", func=None, arg=None):
    start = timer()
    res = func(arg)
    end = timer()
    print(text, " took : ", end-start, " seconds")
    return res 

def main():
    print("Python => main()")
    MAX_LEN = int(5*1e8)
    ar = np.random.rand(MAX_LEN)

    def fLoop(ar):
        s = 0.0
        for i in range(0, len(ar)):
            el = ar[i] 
            s = s + (1-el)/(1+el)
        print("res=", s)
        return s

    def nditerLoop(ar):
        s = 0.0
        for el in np.nditer(ar):
            s = s + (1-el)/(1+el)
        print("res=", s)
        return s

    def listComp(ar):
        x = np.cumsum([(1-el)/(1+el) for el in ar])
        print("res=", x)
        return s


    print("BEGIN")
    print("----------------")
    print(ar)

    print("1)")
    # Value is consumed when passed to native code so in order re-use 'ar' we have to pass it by copy
    arCalc = ar.copy()
    res = examply.runCalc(arCalc)
    print("Nim code took : res.elapsed=", res[0], " ms")
    print("res.value=", res[1])
    pyres = timeMe("ndIter loop", nditerLoop, ar)
    print("pyres=", pyres)

    # timeMe("python loop", fLoop, ar)
    # timeMe("List comprehension", listComp, ar)

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
