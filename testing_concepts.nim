import scinim / experimental
#
import arraymancer / tensor


let t = Tensor[float].init(10)
echo $t

let s = seq[float].init(10)
echo s

let t2 = Tensor[float].linspace(0.0, 10.0, 10)
echo t2
let s2 = seq[float].linspace(0.0, 10.0, 10)
echo s2

template `<-`(typ, call: untyped): untyped =
  call -> typ

echo linspace(0.0, 10.0, 10) -> Tensor
echo linspace[float](0.0, 10.0, 10) -> Tensor
echo linspace(0.0, 10.0, 10) -> seq

echo Tensor <- linspace(0.0, 10.0, 10)
echo seq <- linspace(0.0, 10.0, 10)

import random
var rnd: Rand

import sequtils
block Seq:
  let n = 100
  let v = toSeq(0 ..< n).mapIt(rnd.rand(n))
  echo seq[int] is VectorLikeG[int]
  var (a, b) = histogramImpl(v, int)
  # check if the sum of the bins equals the number of samples
  doAssert sum(a) == n
  ## check that the bin counts are evenly spaced when the data is from
  ## a linear function
  (a, b) = histogramImpl(seq[float].linspace(0, 10, 100), int)
  doAssert a == toSeq(0 ..< a.len).mapIt(10)
block Tensor:
  let n = 100
  let v = toSeq(0 ..< n).toTensor.mapInline(rnd.rand(n))
  var (a, b) = histogramImpl(v, int)
  # check if the sum of the bins equals the number of samples
  doAssert sum(a) == n
  # check that the bin counts are evenly spaced when the data is from
  # a linear function
  (a, b) = histogramImpl(linspace(0, 10, 100), int)
  #doAssert a == toSeq(0 ..< a.len).mapIt(10)
