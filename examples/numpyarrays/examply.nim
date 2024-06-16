import nimpy
import scinim/numpyarrays
import scinim
import std/[times, monotimes, math, sequtils]

template timeIt(name:string, body) =
  let t0 = getMonoTime()
  body
  let sub = getMonoTime() - t0
  # echo(name, " took ", $(sub))
  let elapsed {.inject.} = sub.inMicroseconds() / 1000

proc doStuff[T](el: T) : T {.inline.} =
  result = (1.0-el)/(1.0+el)

proc modArray*(x: NumpyArray[float64]) {.exportpy.} =
  # echo "modArrayInPlace.nim"
  # Example of accessing the buffer directly
  x[0, 0] = 123.0
  x[0, 1] = -5.0

proc parallelForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} =
  result = initNumpyArray[float64](x.shape)
  let
    ur = result.toUnsafeView()
    ux = x.toUnsafeView()

  for i in 0||(x.len-1):
    ur[i] = doStuff ux[i]

proc parallelIndexedForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} =
  result = initNumpyArray[float64](x.shape)

  fuseLoops("parallel for"):
    for i in 0..(x.shape[0]-1):
      for j in 0..(x.shape[1]-1):
        result[i, j] = doStuff x[i, j]

proc normalForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} =
  result = initNumpyArray[float64](x.shape)

  for i in 0..(x.len-1):
    result{i} = doStuff x{i}

proc indexedOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} =
  result = initNumpyArray[float64](x.shape)

  for i in 0..<(x.shape[0]):
    for j in 0..<(x.shape[1]):
      result[i, j] = doStuff x[i, j]

proc runCalc*(x: NumpyArray[float64]) : tuple[elapsed: float64, value: float64] {.exportpy.} =
  var s = 0.0
  timeIt("forLoop"):
    for i in 0..<x.len:
      s += doStuff x{i}
  result = (elapsed:elapsed, value: s)
