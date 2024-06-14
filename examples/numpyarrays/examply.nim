import nimpy
import scinim/numpyarrays
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
  x[0] = 123.0
  x[1] = -5.0

proc parallelForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} = 
  let np = pyImport("numpy")
  result = initNumpyArray[float64](x.shape)

  timeIt("parallelForLoop"):
    for i in 0||(x.len-1):
      result[i] = doStuff x[i]

proc normalForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} = 
  let np = pyImport("numpy")
  result = initNumpyArray[float64](x.shape)

  timeIt("normalForLoop"):
    for i in 0..(x.len-1):
      result[[i]] = doStuff x[[i]]

proc runCalc*(x: NumpyArray[float64]) : tuple[elapsed: float64, value: float64] {.exportpy.} =
  var ux = toUnsafeView(x)
  var s = 0.0
  timeIt("forLoop"):
    for i in 0..<x.len:
      s += doStuff ux[i]
  result = (elapsed:elapsed, value: s)
