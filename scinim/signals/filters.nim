import sequtils
import math
import arraymancer / [tensor, linear_algebra]
import polynumeric
import algorithm

proc savitzkyGolayCoeffs*(windowLength: int, polyOrder: int,
                          deriv = 0, delta = 1.0): Tensor[float] =
  ## Computes the Savitzky-Golay coefficients for a window of length
  ## `windowLength` using polynomials up to order `polyOrder`.
  ##
  ## `deriv` determines the order of the derivative to compute. By default
  ## no derivative is used (`deriv = 0`).
  ##
  ## `delta` only applies for derivatives and describes the sampling spacing
  ## of the data.
  let pos = windowLength div 2
  let rem = windowLength mod 2
  if rem == 0:
    raise newException(ValueError, "Given window length must be odd.")

  let x = arange(-pos.float, (windowLength - pos).float)[windowLength-1..0|-1]
  # compute a Vandermonde matrix for `x` up to polyOrder + 1 (to include polynomial order *up to* polyOrder)
  let A = vandermonde(x, polyOrder + 1).transpose

  var y = zeros[float](polyOrder + 1)
  y[deriv] = fac(deriv).float / (pow(delta, deriv.float))

  let (coeffs, _, _, _) = least_squares_solver(A, y)
  result = coeffs

proc fitEdge(windowStart, windowStop: int,
             interpStart, interpStop: int,
             polyOrder: int,
             data: Tensor[float],
             res: var Tensor[float],
             interpSVG: static bool) =
  ## Fits a polynomial of order `polyOrder` to the `data` within the given window
  ## and applies an interpolation to `res` within half the window.
  ##
  ## The `*Stop` values are taken as exclusive stops.
  ##
  ## If `interpSVG` is true, we perform a linear interpolation between the existing
  ## result of the Savitzky-Golay filter stored in `res` and the polynomial fit. The
  ## interpolation uses fully the polynomial fit at the exact edge and fully the data
  ## at the "inner" edge of the data.
  let winLength = windowStop - windowStart
  let xrange = arange(0.0, winLength.float)          # the x values at which to fit
  let yrange = data[windowStart ..< windowStop]      # the data to fit to
  let polyCoeff = polyFit(xrange, yrange, polyOrder) # perform the fit of desired order
  # NOTE: `initPoly` receives coefficient of polynomial of highest order ``first``!
  let p = initPoly(polyCoeff.toRawSeq.reversed)
  for i in interpStart ..< interpStop:
    when interpSVG:
      let part = block:
        let tmp = (interpStop - i).float / interpStop.float
        if interpStop < windowStop:
          tmp
        else:
          1.0 - tmp
      res[i] = (1.0 - part) * res[i] + part * p.eval(xrange[i - windowStart])
    else:
      # use purely the polynomial fit in the interpolation range
      res[i] = p.eval(xrange[i - windowStart])

proc interpolateEdges(filtered, y: Tensor[float], windowLength: int, polyOrder: int): Tensor[float] =
  ## Performs interpolation of the given SVG `filtered` data using a polynomial fit to
  ## the input data `y` within `windowLength` of order `polyOrder`.
  result = filtered.clone()
  fitEdge(0, windowLength,                 # window range
          0, windowLength div 2,           # interp range
          polyOrder,
          y, result,
          interpSVG = true)
  let num = filtered.size.int
  fitEdge(num - windowLength, num,         # window range
          num - (windowLength div 2), num, # interp range
          polyOrder,
          y, result,
          interpSVG = true)

proc convolve1D*(input: Tensor[float], kernel: Tensor[float]): Tensor[float] =
  ## Convolution of the `input` with the given `kernel`.
  ##
  ## Currently it only allows a convolution including a constant value extension
  ## of the data by 0 (i.e. the convolution is computed over
  ## `[-kernel.size, input.size + kernel.size]`, but the `input` is taken as 0
  ## outside the range of the input).
  ##
  ## Note: The implementation is naive. Performance improvements are certainly
  ## achievable.
  result = newTensor[float](input.size.int)
  let offset = kernel.size div 2
  for i in 0 ..< input.size:
    # compute start and stop of the window
    let windowStart = if i >= offset: 0    # window fully in data range
                      else: offset - i     # part is still outside (essentially extend by 0 data)
    let windowStop  = if i + offset < input.size: kernel.size - 1 # kernel fits fully into rest of data
                      else: input.size - 1 - i + offset           # stop early (extend by 0 data)
    for j in windowStart ..< windowStop:
      result[i] += input[i - offset + j] * kernel[j]

proc savitzkyGolayFilter*(data: Tensor[float], windowLength, polyOrder: int,
                          interpolateEdges = true): Tensor[float] =
  ## Computes the result of applying a Savitzky-Golay filter to the input `data`
  ## using the given `windowLength` and polynomial order `polyOrder`.
  ##
  ## If `interpolateEdges` is true, we will perform a polynomial interpolation
  ## on the edges of the resulting filtered data to accomodate the problem of
  ## bad predictions at the edges of our data, due to extending the data by
  ## zeroes in the convolution.
  ##
  ## Note: this implementation depends on LAPACK, as it uses `gelsd` to perform
  ## linear least squares solving.
  let coeffs = savitzky_golay_coeffs(windowLength, polyOrder)
  result = convolve1D(data, coeffs)
  if interpolateEdges:
    result = result.interpolateEdges(data, windowLength, polyOrder)
