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
  var ux = toUnsafeView(x)
  ux[0] = 123
  ux[1] = -5

proc parallelForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} = 
  let np = pyImport("numpy")
  result = asNumpyArray[float64](np.zeros(x.shape))
  var us = toUnsafeView(result)

  timeIt("parallelForLoop"):
    for i in 0||(x.len-1):
      us[i] = doStuff us[i]

proc normalForOp*(x: NumpyArray[float64]) : NumpyArray[float64] {.exportpy.} = 
  let np = pyImport("numpy")
  result = asNumpyArray[float64](np.zeros(x.shape))
  var us = toUnsafeView(result)

  timeIt("normalForLoop"):
    for i in 0..(x.len-1):
      us[i] = doStuff us[i]


proc runCalc*(x: NumpyArray[float64]) : tuple[elapsed: float64, value: float64] {.exportpy.} =
  # echo "runCalc.nim on ", x.len, " elts"
  var ux = toUnsafeView(x)

  var s = 0.0
  timeIt("forLoop"):
    for i in 0..<x.len:
      s += doStuff ux[i]
  # echo "From nim    -> res=", s

  # echo almostEqual(s, res)
  result = (elapsed:elapsed, value: s)