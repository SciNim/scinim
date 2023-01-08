#import arraymancer / tensor / display
# avoid items and pairs, as they clash with ours
# if imported causes weird concept mismatch errors
#import arraymancer/laser/strided_iteration/foreach
import sequtils
import macros, typetraits
import math

from algorithm import nil
import private / [tensor_utils, macro_utils]

export tensor_utils
export macro_utils

template vectorBody(): untyped =
  ## The concept body that needs to be satisfied such that something is considered to
  ## be `VectorLike`. We require a `len`, access to the indices and iterators via
  ## `iter` and `pairIter`. The default `items` and `pairs` are *not* used on purpose,
  ## to avoid ambiguities (e.g. an arraymancer `Tensor` defines these iterators, but with
  ## a meaning separate from what we require).
  x.len is int
  x[0] is T
  for el in iter(x):
    el is T
  for i, el in pairIter(x):
    i is int
    el is T
  ## XXX: if we use this together with `{.explain.}` the compiler explodes by eating all RAM
  ## Without `{.explain.}` it causes the concept not to match for Tensor anymore
  #x.toUnsafeView() is (ptr UncheckedArray[T])

type
  ScalarLike* {.explain.} = concept x
    x.isScalar() ## maybe there's a better way, but I think for now a user "opt-in" approach to
                 ## a type being a `Scalar` seems like a reasonable idea. For any idea I can
                 ## come up with to check if something is "Scalar", I can think of counter examples
                 ## that don't match this.

  ## An explicit generic version that makes it easy to get the inner types as an explicit
  ## generic argument.
  #VectorLikeG*[T] {.explain.} = concept x
  VectorLikeG*[T] = concept x #, type T
    vectorBody()

  ## An implicitly generic object for use cases where the generic inner type is not needed.
  #VectorLike* {.explain.} = concept x
  VectorLike* = concept x #, type T
    type T = typeof(x[0])
    vectorBody()

proc init*[T: seq](_: typedesc[T], size: int = 0): T =
  result = newSeq[getSubType(T)](size)

proc init*[T: seq; U](_: typedesc[T], inner: typedesc[U], size: int = 0): seq[T] =
  result = seq[U].init()

proc toUnsafeView*[T](s: seq[T]): ptr UncheckedArray[T] =
  result = cast[ptr UncheckedArray[T]](s[0].addr)

proc high*[T: VectorLike](x: T): int = x.len - 1

iterator iter*[T](t: seq[T]): T =
  for i in 0 ..< t.len:
    yield t[i]

iterator miter*[T](t: var seq[T]): var T =
  for i in 0 ..< t.len:
    yield t[i]

iterator pairIter*[T](t: seq[T]): (int, T) =
  for i in 0 ..< t.len:
    yield (i, t[i])

template map_inline*(v: VectorLike, body: untyped): untyped =
  type outType = typeof((
    block:
      var x {.inject.}: type(v[0])
      body
  ))
  type Y = outerType(type(v), outType)
  var res = Y.init(v.len)
  static: echo "OUTTYPE ", typeof(Y), " typeof ", typeof(res)
  for i, x {.inject.} in pairIter(v):
    static: echo "type of ele ", typeof(x)
    static: echo "type of body ", typeof(body)
    res[i] = body
  res

template filter_inline*(v: VectorLike, cond: untyped): untyped =
  var idx = newSeqOfCap[int](v.len)
  for i, x {.inject.} in pairIter(v):
    if cond:
      idx.add i
  var res = init(type(v), idx.len)
  for i, val in pairIter(idx):
    res[i] = v[val]
  res

template apply_inline*(v: var VectorLike, body: untyped): untyped =
  for i, x {.inject.} in pairIter(v):
    v[i] = body

proc sum*[T](s: VectorLikeG[T]): T =
  result = T(0)
  for x in s:
    result += x

proc prod*[T](s: VectorLikeG[T]): T =
  result = T(1)
  for x in s:
    result *= x

proc mean*[T](s: VectorLikeG[T]): T =
  result = s.sum / s.len.T

proc `[]`*[T: VectorLike](a: T, inds: VectorLikeG[int]): T {.inline.} =
  ## given two openArrays, return a sequence of all elements whose indices
  ## are given in 'inds'
  ## inputs:
  ##    a: seq[T] = the sequence from which we take values
  ##    inds: openArray[int] = the array which contains the indices for the
  ##         arrays, which we take from 'array'
  ## outputs:
  ##    seq[T] = a sequence of all elements s.t. array[ind] in numpy indexing
  result = T.init(inds.len)
  for i, ind in pairIter(inds):
    result[i] = a[ind]

proc toSeq*[T: VectorLike](_: type T, sl: Slice[int]): T =
  result = T.init(sl.b - sl.a)
  var i = 0
  for val in sl.a .. sl.b:
    result[i] = val
    inc i

proc sorted*[T: VectorLike](x: T): T =
  ## Returns a sorted version of the given `x`
  result = x
  ## XXX: this does not work for some reason
  algorithm.sort(toOpenArray(result.toUnsafeView(),
                             0,
                             result.high))

## For some reason the compiler is too dumb for this
# proc zip*[T: VectorLike; U: VectorLike](a: T, b: U): auto =
#  type Tup = (getSubType(T), getSubType(U))
proc zip*[T; U](a: VectorLikeG[T], b: VectorLikeG[U]): auto =
  type Tup = tuple[a: T, b: U]
  doAssert a.len == b.len
  result = outerType(type(a), Tup).init(a.len)
  for i in 0 ..< a.len:
    ## XXX: `cast` because of compiler bug...
    result[i] = cast[Tup]((a: a[i], b: b[i]))

template liftCompareProc(op) =
  ## lift an comparator operator like `<` to work element wise
  ## on two openArrays `x`, `y` and return a `seq[bool]`
  proc `op`*[T](x, y: VectorLikeG[T]): VectorLikeG[bool] =
    result = outerType(type(x), bool).init(x.len)
    for (i, xy) in pairIter(zip(x, y)):
      result[i] = op(xy.a, xy.b)

# lift comparator operators
liftCompareProc(`<`)
liftCompareProc(`>`)
liftCompareProc(`<=`)
liftCompareProc(`>=`)

proc linspace*[T: VectorLike](_: type T, start, stop: float,
                              num: int, endpoint = true): T =
  ## linspace similar to numpy's linspace
  ## returns a seq containing a linear spacing starting from `start` to `stop`
  ## either including (endpoint == true) or excluding (endpoint == false) `stop`
  ## with a number of `num` elements
  var
    step = start
    diff: float
  if endpoint == true:
    diff = (stop - start) / float(num - 1)
  else:
    diff = (stop - start) / float(num)
  if diff < 0:
    # in case start is bigger than stop, return an empty sequence
    return init(T, 0)
  else:
    result = init(T, num)
    for i in 0 ..< num:
      result[i] = start + i.float * diff

proc linspaceT*(start, stop: float, num: int, endpoint = true): Tensor[float] =
  ## linspace similar to numpy's linspace
  ## returns a `Tensor` containing a linear spacing starting from `start` to `stop`
  ## either including (endpoint == true) or excluding (endpoint == false) `stop`
  ## with a number of `num` elements
  result = Tensor[float].linspace(start, stop, num, endpoint)

proc linspaceS*(start, stop: float, num: int, endpoint = true): seq[float] =
  ## linspace similar to numpy's linspace
  ## returns a `seq` containing a linear spacing starting from `start` to `stop`
  ## either including (endpoint == true) or excluding (endpoint == false) `stop`
  ## with a number of `num` elements
  result = seq[float].linspace(start, stop, num, endpoint)

proc arange*[T: VectorLike](_: type T, start, stop, step = 1, endpoint = false): T =
  ## returns seq containing all elements from incl. `start` to excl. `stop`
  ## given a stepping of `step`
  ## `endpoint` allows to include `stop` in the output array.
  ## Similar to Numpy's arange
  var mstop = stop
  if endpoint == true:
    mstop = stop + 1
  let nElems = ((mstop - start + 1).float / step.float).round.int
  result = init(T, nElems)
  var
    i = 0
    val = start
  while val < mstop:
    result[i] = val
    inc val, step
    inc i

proc arangeS*(start, stop, step = 1, endpoint = false): seq[int] =
  result = seq[int].arange(start, stop, step, endpoint)

proc arangeT*(start, stop, step = 1, endpoint = false): Tensor[int] =
  result = Tensor[int].arange(start, stop, step, endpoint)

proc max*[T](s: VectorLikeG[T]): T =
  if s.len == 0: return T(0)
  result = s[0]
  for x in iter(s):
    result = max(x, result)

proc min*[T](s: VectorLikeG[T]): T =
  if s.len == 0: return T(0)
  result = s[0]
  for x in iter(s):
    result = min(x, result)

proc cumProd*[T](x: VectorLikeG[T]): VectorLike =
  ## cumulative product for each element of ``x``
  ##
  ## ``cumProd(@[1,2,3,4])`` produces ``@[1,2,6,24]``
  result = init(type(x), x.len)
  var cp = T(1)
  for i in 0 ..< x.len:
    cp = cp * x[i]
    result[i] = cp

proc cumSum*[T](x: VectorLikeG[T]): VectorLike =
  ## cumulative sum for each element of ``x``
  ##
  ## ``cumSum(@[1,2,3,4])`` produces ``@[1,3,6,10]``
  result = init(type(x), x.len)
  var cp = T(0)
  for i in 0 ..< x.len:
    cp = cp + x[i]
    result[i] = cp

proc cumCount*[T: SomeInteger](x: VectorLikeG[T], v: T): VectorLike =
  ## cumulative count of a number in ``x``
  ##
  ## the cumulative count of ``3`` for ``@[1,3,3,2,3]`` produces ``@[0,1,2,2,3]``
  result = init(type(x), x.len)
  var cp = T(0)
  for i in 0 ..< x.len:
    if x[i] == v: inc(cp)
    result[i] = cp

proc cumPowSum*[T](x: VectorLike, p: T): VectorLike =
  ## cumulative sum of ``pow(x[], p)`` for each element
  ## The resultant sequence is of type ``float``
  ##
  ## ``cumPowSum([1,2,3,4],2)`` produces ``@[1, 5, 14, 30]``
  result = type(x).init(x.len)
  var cps = 0.0
  for i in 0 ..< x.len:
    cps += pow(x[i].toFloat, p.toFloat)
    result[i] = cps

# ----------- single-result seq math -----------------------

proc product*[T](x: VectorLikeG[T]): T =
  ## sum each element of ``x``
  ## returning a single value
  ##
  ## ``product(@[1,2,3,4])`` produces ``24`` (= 1 * 2 * 3 * 4)
  var cp = T(1)
  for i in 0 ..< x.len: cp *= x[i]
  result = cp

proc sumSquares*[T](x: VectorLikeG[T]): T =
  ## sum of ``x[i] * x[i]`` for each element
  ## returning a single value
  ##
  ## ``sumSquares(@[1,2,3,4])``
  ## produces ``30``  (= 1*1 + 2*2 + 3*3 + 4*4)
  var ps = T(0)
  for i in iter(x):
    ps += i*i
  result = ps

proc powSum*[T](x: VectorLikeG[T], p: T): float =
  ## sum of ``pow(x[], p)`` of each element
  ## returning a single value
  ##
  ## ``powSum(@[1,2], 3)``
  ## produces ``9``  (= pow(1,3) + pow(2,3))
  var ps = 0.0
  for i in 0 ..< x.len: ps += pow(x[i].toFloat, p.toFloat)
  result = ps

proc max*[T](x: VectorLikeG[T], m: T): VectorLike =
  ## Maximum of each element of ``x`` compared to the value ``m``
  ## as a sequence
  ##
  ## ``max(@[-1,-2,3,4], 0)`` produces ``@[0,0,3,4]``
  if x.len == 0: result = @[m]
  else:
    result = init(type(x), x.len)
    for i in 0 ..< x.len:
      result[i] = max(m, x[i])

proc argmax*(x: VectorLike): int =
  let m = max(x)
  for i, el in pairIter(x):
    if el == m:
      return i

proc argmin*(x: VectorLike): int =
  let m = min(x)
  for i, el in pairIter(x):
    if el == m:
      return i

proc max*(x, y: VectorLike): VectorLike =
  ## Note: previous definition using an VectorLike as the type
  ## does not work anymore, since it clashes with with
  ## system.max[T](x, y: T) now

  ## Maximum value of each element of ``x`` and
  ## ``y`` respectively, as a sequence.
  ##
  ## ``max(@[-1,-2,3,4], @[4,3,2,1])`` produces ``@[4,3,3,4]``
  if x.len == 0: result = @[]
  else:
    result = init(type(x), x.len)
    let xLen = max(x.len, y.len)
    let nlen = min(x.len, y.len)
    for i in 0 ..< xLen:
      if i < nlen: result[i] = max(x[i], y[i])
      elif i < x.len: result[i] = x[i]
      else: result[i] = y[i]

proc min*[T](x: VectorLikeG[T], m: T): VectorLike =
  ## Minimum of each element of ``x`` compared to the value ``m``
  ## as a sequence
  ##
  ## ``min(@[1,2,30,40], 10)`` produces ``@[1,2,10,10]``
  if x.len == 0: result = @[m]
  else:
    result = init(type(x), x.len)
    for i in 0 ..< x.len:
      result[i] = min(m, x[i])

proc min*(x, y: VectorLike): VectorLike =
  ## Note: previous definition using an VectorLike as the type
  ## does not work anymore, since it clashes with with
  ## system.min[T](x, y: T) now

  ## Minimum value of each element of ``x`` and
  ## ``y`` respectively, as a sequence.
  ##
  ## ``min(@[-1,-2,3,4], @[4,3,2,1])`` produces ``@[-1,-2,2,1]``
  if x.len == 0: result = init(type(x), x.len)
  else:
    result = init(type(x), x.len)
    let xLen = max(x.len, y.len)
    let nlen = min(x.len, y.len)
    for i in 0 ..< xLen:
      if i < nlen: result[i] = min(x[i], y[i])
      elif i < x.len: result[i] = x[i]
      else: result[i] = y[i]

proc bincount*(x: VectorLike, minLength: int): VectorLikeG[int] =
  ## Count of the number of occurrences of each value in
  ## sequence ``x`` of non-negative ints.
  ##
  ## The result is a sequence of length ``max(x)+1``
  ## or ``minLength`` if it is larger than ``max(x)``.
  ## Covering every integer from ``0`` to
  ## ``max(max(x), minLength)``
  doAssert min(x) >= 0, "Negative values are not allowed in bincount!"
  let size = max(max(x) + 1, minLength)
  static: echo "TYPE OF ISS ??? ", type(x)
  result = init(type(x), size)
  for idx in iter(x):
    inc(result[idx])

proc bincount*(x: VectorLikeG[int], minLength: int,
               weights: VectorLike): VectorLikeG[int] =
  ## version of `bincount` taking into account weights. The resulting dtype is
  ## the type of the given weights.
  doAssert min(x) >= 0, "Negative values are not allowed in bincount!"
  let size = max(max(x) + 1, minLength)
  result = init(type(x), size)
  doAssert weights.len == x.len or weights.len == 0
  if weights.len > 0:
    for wIdx, rIdx in pairIter(x):
      result[rIdx] += weights[wIdx]
  else:
    for wIdx, rIdx in pairIter(x):
      result[rIdx] += 1

proc digitize*(x: VectorLike, bins: VectorLike, right = false): VectorLikeG[int] =
  ## Return the indices of the ``bins`` to which each value of ``x`` belongs.
  ##
  ## Each returned index for *increasing ``bins``* is ``bins[i-1]<=x< bins[i]``
  ## and if ``right`` is true, then returns ``bins[i-1]<x<=bins[i]``
  ##
  ## Each returned index for *decreasing ``bins``* is ``bins[i-1] > x >= bins[i]``
  ## and if ``right`` is true, then returns ``bins[i-1] >= x > bins[i]``
  ##
  ## Note: if ``x`` has values outside of ``bins``, then ``digitize`` returns an index
  ## outside the range of ``bins`` (``0`` or ``bins.len``)
  doAssert(bins.len > 1,"digitize() must have two or more bin values")
  result = init[VectorLike[int]](x.len)
  # default of increasing bin values
  for i in 0 ..< x.len:
    result[i] = bins.high + 1
    if bins[1] > bins[0]:
      for k in 0 ..< bins.len:
        if x[i] < bins[k] and not right:
          result[i] = k
          break
        elif x[i] <= bins[k] and right:
          result[i] = k
          break
    #decreasing bin values
    else:
      for k in 0 ..< bins.len:
        if x[i] >= bins[k] and not right:
          result[i] = k
          break
        elif x[i] > bins[k] and right:
          result[i] = k
          break

func areBinsUniform(bin_edges: VectorLikeG[float]): bool =
  ## simple check whether bins are uniform
  if bin_edges.len in {0, 1, 2}: return true
  else:
    let binWidth = bin_edges[1] - bin_edges[0]
    for i in 0 ..< bin_edges.high:
      if abs((bin_edges[i+1] - bin_edges[i]) - binWidth) > 1e-8:
        return false

proc rebin*[T](bins: VectorLikeG[T], by: int): VectorLike =
  ## Given a set of `bins`, `rebin` combines each consecutive `by` bins
  ## into a single bin, summing the bin contents. The resulting seq thus
  ## has a length of `bins.lev div by`.
  ## TODO: add tests for this!
  result = init(type(bins), bins.len div by)
  var tmp = T(0)
  var j = 0
  for i in 0 .. bins.high:
    if i > 0 and i mod by == 0:
      result[j] = tmp
      tmp = T(0)
      inc j
    tmp += bins[i]

proc fillHist*[T; U: VectorLike](_: typedesc[T],
                     #bins: VectorLikeG[T], data: VectorLikeG[T],
                     bins: U, data: U,
                     upperRangeBinRight = true): T =
  ## Given a set of `bins` (which are interpreted to correspond to the left
  ## bin edge!) and a sequence of `data`, it will fill a histogram according
  ## to the `bins`. That is for each element `d` of `data` the correct bin
  ## `b` is checked and this bin is increased by `1`.
  ## If `upperRangeBinRight` is true, the last bin entry is considered the right
  ## edge of the last bin. All values larger than it will be dropped. Otherwise
  ## the last bin includes everything larger than bins[^1].
  ## TODO: write tests for this!
  var mbins = bins
  if not upperRangeBinRight:
    # add `inf` bin to `mbins` as upper range
    mbins.add Inf
  result = outerType(type(bins), int).init(mbins.len - 1)
  let dataSorted = data.sorted
  var
    curIdx = 0
    curBin = mbins[curIdx]
    nextBin = mbins[curIdx + 1]
    idx = 0
    d: getSubType(type(dataSorted))
  while idx < dataSorted.len:
    d = dataSorted[idx]
    if d >= curBin and d < nextBin:
      inc result[curIdx]
      inc idx
    elif d < curBin:
      # outside of range
      inc idx
    elif d >= nextBin:
      inc curIdx
      if curIdx + 1 == mbins.len: break
      curBin = mbins[curIdx]
      nextBin = mbins[curIdx + 1]
    else:
      doAssert false, "should never happen!"

type
  Missing* = object
  W* = Missing | VectorLike

proc missing*(): Missing = discard

proc histogramImpl*[T: int|float; U: float | int; W](
  arg: VectorLikeG[T],
  dtype: typedesc[U],
  bins: (int | string | VectorLike) = 10,
  range: tuple[mn, mx: float] = (0.0, 0.0),
  weights: W = missing(),
  density: static bool = false,
  upperRangeBinRight = true): auto = #(genericHead(type(x))[dtype],
    #genericHead(type(x))[float]) =
  #(VectorLikeG[dtype], VectorLikeG[float]) =
  ## Compute the histogram of a set of data. Adapted from Numpy's code.
  ## If `bins` is an integer, the required bin edges will be calculated in the
  ## range `range`. If no `range` is given, the `(min, max)` of `x` will be taken.
  ## If `bins` is a `seq[T]`, the bin edges are taken as is. Note however, that
  ## the bin edges must include both the left most, as well as the right most
  ## bin edge. Thus the length must be `numBins + 1` relative to the desired number
  ## of bins of the resulting histogram!
  ## The behavior of range can be set via `upperRangeBinRight`. It controls the
  ## interpretation of the upper range. If it is `true` the upper range is considered
  ## the right edge of the last bin. If `false` we understand it as the left bin edge
  ## of the last bin. This of course only has an effect if `bins` is given as an
  ## integer!
  ## Returns a tuple of
  ## - histogram: seq[dtype] = the resulting histogram binned via `bin_edges`. `dtype`
  ##     is `int` for unweighted histograms and `float` for float weighted histograms
  ## - bin_edges: seq[T] = the bin edges used to create the histogram


  ## XXX: possibly rewrite to use `seq` internally for some things (e.g. `fillHist` requires
  ## `add` calls) and convert to correct type at the end?
  type X = outerType(type(arg), float)
  if arg.len == 0:
    raise newException(ValueError, "Cannot compute histogram of empty array!")

  when W isnot Missing:
    if weights.len > 0 and weights.len != arg.len:
      raise newException(ValueError, "The number of weights needs to be equal to the number of elements in the input seq!")
  var uniformBins = true # are bins uniform?

  # parse the range parameter
  var (mn, mx) = range
  if classify(mn) == fcNan or classify(mx) == fcNaN:
    raise newException(ValueError, "One of the input ranges is NaN!")
  if classify(mn) in {fcInf, fcNegInf} or classify(mx) in {fcInf, fcNegInf}:
    raise newException(ValueError, "One of the input ranges is Inf!")

  if mn == 0.0 and mx == 0.0:
    mn = arg.min.float
    mx = arg.max.float
  if mn > mx:
    raise newException(ValueError, "Max range must be larger than min range!")
  elif mn == mx:
    mn -= 0.5
    mx += 0.5
  # from here on mn, mx unchanged
  when type(bins) is string:
    # to be implemented to guess the number of bins from different algorithms. Looking at the Numpy code
    # for the implementations it's only a few LoC
    raise newException(NotImplementedError, "Automatic choice of number of bins based on different " &
                       "algorithms not implemented yet.")
  elif type(bins) is VectorLike:
    let bin_edges = bins.map_inline(x.float)
    let numBins = bin_edges.len - 1
    # possibly truncate the input range (e.g. bin edges smaller range than data)
    mn = min(bin_edges[0], mn)
    mx = min(bin_edges[^1], mx)
    # check if bins really uniform
    uniformBins = areBinsUniform(bin_edges)
  elif type(bins) is int:
    if bins == 0:
      raise newException(ValueError, "0 bins is not a valid number of bins!")
    let numBins = bins
    var bin_edges: X
    if upperRangeBinRight:
      bin_edges = X.linspace(mn, mx, numBins + 1, endpoint = true)
    else:
      let binWidth = (mx - mn) / (numBins.float - 1)
      bin_edges = X.linspace(mn, mx + binWidth, numBins + 1, endpoint = true)

  when T isnot float:
    var x_data = arg.map_inline(x.float)
    # redefine locally as floats
    when W isnot Missing:
      let w: VectorLikeG[T] = weights
      var weights = w.map_inline(x.float)
    else:
      var weights = outerType(type(arg), float).init(0)
  else:
    var x_data = arg
    # weights already float too, redefine mutable
    when W isnot Missing:
      var weights: VectorLikeG[float] = weights
    else:
      var weights = outerType(type(arg), float).init(0)

  if uniformBins:
    # normalization
    let norm = numBins.float / (mx - mn)
    # make sure input array is float and filter to all elements inside valid range
    # x_keep is used to calculate the indices whereas x_data is used for hist calc
    let idxData = outerType(type(x_data), int).toSeq(0 ..< x_data.len)
    let idxKeep = idxData.filter_inline(x_data[x] >= mn and x_data[x] <= mx)
    var x_keep = idxKeep.map_inline(x_data[x])
    x_data = x_keep
    # apply to weights if any
    if weights.len > 0:
      weights = idxKeep.map_inline(weights[x])
    # remove potential offset
    for x in miter(x_keep):
      x = (x - mn) * norm

    # compute bin indices
    var indices = map_inline(x_keep, x.int)
    # for indices which are equal to the max value, subtract 1
    indices.apply_inline:
      if x == numBins:
        x - 1
      else:
        x
    # since index computation not guaranteed to give exactly consistent results within
    # ~1 ULP of the bin edges, decrement some indices
    let decrement = x_data < bin_edges[indices]
    for i in 0 .. indices.high:
      if decrement[i] == true:
        indices[i] -= 1
      if x_data[i] >= bin_edges[indices[i] + 1] and indices[i] != (numBins - 1):
        indices[i] += 1
    # currently weights and min length not implemented for bincount
    when dtype is int:
      result = (bincount(indices, minLength = numBins), bin_edges)
    else:
      result = (bincount(indices, minLength = numBins,
                         weights = weights),
                bin_edges)
  else:
    # bins are not uniform
    doAssert weights.len == 0, "Weigths are currently unsupported for histograms with " &
      "unequal bin widths!"
    static: echo "TYPE TYPE TYPE ", typeof(bin_edges), " and ", typeof(x_data)
    let hist = outerType(type(arg), int).fillHist(bin_edges, x_data,
                                                  upperRangeBinRight = upperRangeBinRight)
    when dtype is float:
      result = (hist.map_inline(x.float),
                bin_edges)
    else:
      result = (hist,
                bin_edges)
  when dtype is float:
    if density:
      # normalize the result
      let tot = result[0].sum
      for i in 0 ..< bin_edges.high:
        result[0][i] = result[0][i] / (bin_edges[i+1] - bin_edges[i]) / tot
  else:
    if density:
      raise newException(ValueError, "Cannot satisfy `density == true` with a " &
        "dtype of `int`!")

proc histogram*[T](
  x: T,
  bins: (int | string | VectorLike) = 10,
  range: tuple[mn, mx: float] = (0.0, 0.0),
  density: static bool = false,
  upperRangeBinRight = true,
  dtype: typedesc = int): (auto, auto) = # VectorLikeG[float]) =
  ## Computes the histogram of `x` given `bins` in the desired
  ## `range`.
  ## Right most bin edge by default is assumed to be right most
  ## bin edge, can be changed via `upperRangeBinRight`. If weights
  ## are desired, see the `histogram` overload below. For a more
  ## detailed docstring see `histogramImpl`.
  ##
  ## `density` is a static argument, because we a density normalized
  ## histogram returns float values, whereas a normal histogram is
  ## a sequence of ints.
  when density:
    # when density is to be returned, result must be float
    type dtype = float
  else:
    type dtype = int

  result = histogramImpl(x = x,
                         dtype = dtype,
                         bins = bins,
                         range = range,
                         density = density,
                         upperRangeBinRight = upperRangeBinRight)

#proc histogram*[T; U: float | int](
#  x: VectorLike,
#  weights: VectorLikeG[U],
#  bins: (int | string | VectorLike) = 10,
#  range: tuple[mn, mx: float] = (0.0, 0.0),
#  density: static bool = false,
#  upperRangeBinRight = true): (VectorLikeG[U], VectorLikeG[float]) =
#  ## Computes the histogram of `x` given `bins` in the desired
#  ## `range`.
#  ## Right most bin edge by default is assumed to be right most
#  ## bin edge, can be changed via `upperRangeBinRight`. If weights
#  ## are not desired, see the `histogram` overload above. For a more
#  ## detailed docstring see `histogramImpl`.
#  ##
#  ## `density` is a static argument, because we a density normalized
#  ## histogram returns float values, whereas a normal histogram is
#  ## a sequence of ints.
#  when density:
#    type dtype = float
#  else:
#    type dtype = U
#  result = histogramImpl(x = x,
#                         dtype = dtype,
#                         bins = bins,
#                         range = range,
#                         weights = weights,
#                         density = density,
#                         upperRangeBinRight = upperRangeBinRight)
