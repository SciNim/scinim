import arraymancer

import nimpy
import ../scinim/numpyarrays
import unittest

when defined(osx):
  import nimpy/py_lib as lib
  pyInitLibPath("/Users/regis.caillaud/.pyenv/versions/3.11.9/lib/libpython3.11.dylib")

proc test(arg: tuple[s: string]) =
  suite arg.s:
    test "int":
      var A: Tensor[int64] = toTensor(@[[1'i64, 2, 3], [4'i64, 5, 6]])
      var pA = toNdArray(A)
      pyprint(pA.dtype())
      check toTensor[int64](pA) == A

    test "float":
      var A: Tensor[float64] = toTensor(@[[1.1'f64, 2.2, 3.3], [4.4'f64, 5.5, 6.6]])
      var pA = toNdArray(A)
      pyprint(pA.dtype())
      check toTensor[float](pA) == A

    test "int32":
      var A: Tensor[int32] = toTensor(@[[1'i32, 2, 3], [4'i32, 5, 6]])
      var pA = toNdArray(A)
      pyprint(pA.dtype())
      check toTensor[int32](pA) == A

    test "float32":
      var A: Tensor[float32] = toTensor(@[[1.1'f32, 2.2, 3.3], [4.4'f32, 5.5, 6.6]])
      var pA = toNdArray(A)
      pyprint(pA.dtype())
      check toTensor[float32](pA) == A

    test "RaiseAssert":
      let np = pyImport("numpy")
      let py_array_type = dtype(float32)
      let pA = nimpy.callMethod(np, "zeros", @[2, 3, 4], py_array_type)
      pyprint(pA.dtype())

      expect AssertionDefect:
        var ppA = asNumpyArray[float64](pA)
        pyprint(ppA.dtype())
      check true

    test "RaiseAssert from double object":
      var A: Tensor[float32] = toTensor(@[[1.1'f32, 2.2, 3.3], [4.4'f32, 5.5, 6.6]])
      var pA = toNdArray(A).obj() # Create a PyObject. In practice, this will often be the result of a callMethod proc
      pyprint(pA.dtype())
      expect AssertionDefect:
        var ppA = asNumpyArray[float64](pA)
        pyprint(ppA.dtype())
      check true

    test "Call a Python function":
      var A: Tensor[float32] = toTensor(@[[1.1'f32, 2.2, 3.3], [4.4'f32, 5.5, 6.6]])
      var pA = toNdArray(A)
      pyprint(pA)
      let np = pyImport("numpy")
      # This effectively perform a copy because np.transpose is not C contiguous
      let ret = asNumpyArray[float32](np.transpose(pA))
      pyprint(ret)
      check ret.toTensor() == A.transpose()

    test "Call a Python function using a compistion of NumpyArray":
      var A: Tensor[float32] = toTensor(@[[1.1'f32, 2.2, 3.3], [4.4'f32, 5.5, 6.6]])
      var pA = toNdArray(A)
      var B : Tensor[float32] = toTensor(@[[1.1'f32, 2.2, 3.3], [4.4'f32, 5.5, 6.6]])
      var pB = toNdArray(B)
      let py = pyBuiltinsModule()
      discard nimpy.callMethod(py, "print", (a: pA, b: pB))


when isMainModule:
  test((s: "toTensor, toNdArray in main thread"))
  # Disable for now
  # var thr: Thread[tuple[s: string]]
  # createThread(thr, test, (s: "toTensor, toNdArray in external thread"))
  # joinThread(thr)
