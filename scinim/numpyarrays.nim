import std/sequtils
import std/strformat
import std/tables

import fusion/smartptrs

import arraymancer

import nimpy
import nimpy/[raw_buffers, py_types]

{.push gcsafe.}

proc dtype*(t: PyObject): PyObject =
  nimpy.getAttr(t, "dtype")

proc pyprint*(o: PyObject) =
  let py = pyBuiltinsModule()
  discard nimpy.callMethod(py, "print", o)

proc nptypes(name: string): PyObject =
  let np = pyImport("numpy")
  nimpy.getAttr(np, name)

template dtype*(T: typedesc[int8]): PyObject = nptypes("byte")
template dtype*(T: typedesc[int16]): PyObject = nptypes("short")
template dtype*(T: typedesc[int32]): PyObject = nptypes("int32")
template dtype*(T: typedesc[int64]): PyObject = nptypes("int64")

template dtype*(T: typedesc[uint8]): PyObject = nptypes("ubyte")
template dtype*(T: typedesc[uint16]): PyObject = nptypes("ushort")
template dtype*(T: typedesc[uint32]): PyObject = nptypes("uint32")
template dtype*(T: typedesc[uint64]): PyObject = nptypes("uint64")

proc dtype*(T: typedesc[int]): PyObject =
  when sizeof(T) == sizeof(int64):
    dtype(int64)
  elif sizeof(T) == sizeof(int32):
    dtype(int32)
  else:
    {.error: "Unsupported sizeof(uint)".}

proc dtype*(T: typedesc[uint]): PyObject =
  when sizeof(T) == sizeof(uint64):
    dtype(uint64)
  elif sizeof(T) == sizeof(uint32):
    dtype(uint32)
  else:
    {.error: "Unsupported sizeof(uint)".}

proc dtype*(T: typedesc[bool]): PyObject = nptypes("bool")
proc dtype*(T: typedesc[char]): PyObject = nptypes("char")
proc dtype*(T: typedesc[float32]): PyObject = nptypes("single")
proc dtype*(T: typedesc[float64]): PyObject = nptypes("double")
proc dtype*(T: typedesc[Complex32]): PyObject = nptypes("csingle")
proc dtype*(T: typedesc[Complex64]): PyObject = nptypes("cdouble")

proc assertNumpyType[T](ndArray: PyObject) =
  let
    dtype_sizeof = dtype(ndArray).itemsize.to(int)*sizeof(byte)
    dtype_kind = dtype(ndArray).kind.to(string)[0]

  if sizeof(T) != dtype_sizeof:
    raiseAssert(&"Error converting PyObject NDArray to Arraymancer Tensor. Type sizeof({$T})={sizeof(T)} not equal to numpy.dtype.itemsize ({dtype_sizeof}).")

  let msg = &"Error converting PyObject NDArray to Arraymancer Tensor. Type {$T} not compatible with numpy.dtype.kind {dtype_kind}."
  when T is SomeFloat:
    if dtype_kind != 'f':
      raiseAssert(msg)

  elif T is SomeSignedInt:
    if dtype_kind != 'i':
      raiseAssert(msg)

  elif T is SomeUnsignedInt:
    if dtype_kind != 'u':
      raiseAssert(msg)

  elif T is bool:
    if dtype_kind != 'b':
      raiseAssert(msg)

  else:
    raiseAssert(msg)

type
  PyBuffer = object
    raw: RawPyBuffer

  NumpyArray*[T] = object
    # pyBuf: ptr RawPyBuffer # to keep track of the buffer so that we can release it
    # pyBuf: SharedPtr[RawPyBuffer] # to keep track of the buffer so that we can release it
    pyBuf: SharedPtr[PyBuffer] # to keep track of the buffer so that we can release it
    data*: ptr UncheckedArray[T] # this will be the raw data
    shape*: seq[int]
    len*: int

proc release*(b: var PyBuffer) =
  b.raw.release()

proc `=destroy`*(b: var PyBuffer) =
  b.release()

proc raw(x: SharedPtr[PyBuffer]): RawPyBuffer =
  x[].raw

proc raw(x: var SharedPtr[PyBuffer]): var RawPyBuffer =
  x[].raw

proc obj*[T](x: NumpyArray[T]): PyObject =
  pyValueToNim(x.pyBuf.raw.obj, result)

proc dtype*[T](ar: NumpyArray[T]): PyObject =
  return dtype(T)

proc nimValueToPy*[T](v: NumpyArray[T]): PPyObject {.inline.} =
  nimValueToPy(v.obj())

proc pyprint*[T](ar: NumpyArray[T]) =
  ## Short cut to call print() on a NumpyArray
  let py = pyBuiltinsModule()
  discard nimpy.callMethod(py, "print", ar)

proc initNumpyArray*[T](ar: sink PyObject): NumpyArray[T] =
  result.pyBuf = newSharedPtr(PyBuffer())
  ar.getBuffer(result.pyBuf.raw, PyBUF_WRITABLE or PyBUF_ND)
  let shapear = cast[ptr UncheckedArray[Py_ssize_t]](result.pyBuf.raw.shape)
  for i in 0 ..< result.pyBuf.raw.ndim:
    let dimsize = shapear[i].int # py_ssize_t is csize
    result.shape.add dimsize
  result.len = result.shape.foldl(a * b, 1)
  result.data = cast[ptr UncheckedArray[T]](result.pyBuf.raw.buf)

proc asNumpyArray*[T](ar: sink PyObject): NumpyArray[T] =
  ## some PyObject that points to a numpy array
  ## User has to make sure that the data type of the array can be
  ## cast to `T` without loss of information!
  assertNumpyType[T](ar)
  if not ar.data.c_contiguous.to(bool):
    let np = pyImport("numpy")
    var ar = np.ascontiguousarray(ar)
    return initNumpyArray[T](ar)
  else:
    return initNumpyArray[T](ar)

proc ndArrayFromPtr*[T](t: ptr T, shape: seq[int]): NumpyArray[T] =
  let np = pyImport("numpy")
  let py_array_type = dtype(T)
  # Just a trick to force an initialization of a Numpy Array of the correct size
  result = asNumpyArray[T](
    nimpy.callMethod(np, "zeros", shape, py_array_type)
  )
  # copyMem the data
  var bsizes = result.len*(sizeof(T) div sizeof(uint8))
  copyMem(addr(result.data[0]), t, bsizes)

proc toUnsafeView*[T](ndArray: NumpyArray[T]): ptr UncheckedArray[T] =
  ndArray.data

# Arraymancer only
proc numpyArrayToTensor[T](ndArray: NumpyArray[T]): Tensor[T] =
  result = newTensor[T](ndArray.shape)
  var buf = cast[ptr T](toUnsafeView(ndArray))
  copyFromRaw(result, buf, ndArray.len)

proc toTensor*[T](ndArray: NumpyArray[T]): Tensor[T] =
  result = numpyArrayToTensor[T](ndArray)

proc toTensor*[T](pyobj: PyObject): Tensor[T] =
  var ndArray = asNumpyArray[T](pyobj)
  result = numpyArrayToTensor[T](ndArray)

# Convert Tensor to RawPyBuffer
proc ndArrayFromTensor[T](t: Tensor[T]): NumpyArray[T] =
  # Reshape PyObject to Arraymancer Tensor
  let np = pyImport("numpy")
  var shape = t.shape.toSeq()
  var t = asContiguous(t, rowMajor)
  var buf = cast[ptr T](toUnsafeView(t))
  result = ndArrayFromPtr[T](buf, shape)

proc toNdArray*[T](t: Tensor[T]): NumpyArray[T] =
  ndArrayFromTensor[T](t)

proc pyValueToNim*[T](ar: NumpyArray[T], o: var Tensor[T]) {.inline.} =
  o = toTensor(ar)

{.pop.}
