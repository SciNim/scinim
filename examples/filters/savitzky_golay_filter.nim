import random
import numericalnim / interpolate
import ggplotnim
import arraymancer
import nimpy
import scinim / signals
import sequtils

#[

A simple example computing a smoothed function for some very noisy data
using Savitzky-Golay filters.

We compare the output to Scipy's result (so you need both `nimpy` and
the `scipy` package in your PATH). Finally, the computation requires
LAPACK.

`ggplotnim` is used to plot the data and result.

]#

proc generateData(): seq[float] =
  ## generate some x/y data that has significant noise
  var rng = initRand(123)
  let support = @[1.0, 2.0, 5.0, 2.0, 4.0]
  let at = arange(support.len).asType(float).toRawSeq
  let stds = @[0.5, 1.1, 0.7, 1.7, 0.3]
  let posSpline = newCubicSpline(at, support)
  let stdSpline = newCubicSpline(at, stds)
  const npt = 1000
  result = newSeq[float](npt)
  for i in 0 ..< npt:
    let x = i.float / npt.float * support.high.float
    result[i] = rng.gauss(posSpline.eval(x),
                          stdSpline.eval(x))

# just import the savitzky golay module from scipy
let svg = pyImport("scipy.signal._savitzky_golay")

# generate our random data
let y = generateData().toTensor
let x = toSeq(0 ..< y.len)
# define a window length and polynomial order (the longer the window, the smoother the result)
let windowLength = 889
let polyOrder = 5

# compute 3 different cases
# 1. SVG filter without interpolating on the sides
let filtered = savitzkyGolayFilter(y, windowLength, polyOrder, interpolateEdges = false)
# 2. SVG filter including interpolation on the sides
let finishd = savitzkyGolayFilter(y, windowLength, polyOrder)
# 3. SVG filter using scipy
let purepypy = svg.savgol_filter(y.toRawSeq, windowLength, polyOrder)

var purePy = newSeq[float]()
for x in purepypy:
  purepy.add x.to(float)

let df = seqsToDf(x, y, filtered, purepy, finishd)
ggplot(df, aes("x", "y")) +
  geom_line() +
  geom_line(aes = aes(y = "purepy"), color = some(parseHex("FF00FF"))) +
  geom_line(aes = aes(y = "filtered"), color = some(parseHex("FF0000"))) +
  geom_line(aes = aes(y = "finishd"), color = some(parseHex("0000FF")), size = some(2.0)) +
  ggtitle("Comparison of SVG filters using no interpolation, our interpolation & scipy") +
  ggsave("svg_comparisons.png", width = 1000, height = 800)
