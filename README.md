# scinim
The core types and functions of the SciNim ecosystem

## TODO

- merge with `sugar` branch
- make `sugar` use the concepts
- implement equivalent to `sequtils` procedures based on concepts both
  for general usability & to show concept based `sequtils` works
  - allow consuming iterators in all `sequtils` like templates


## Vector / Scalar features we need

- sorting: can this work properly? At least if we allow copy, but
  without copy we are in trouble with views (e.g. a tensor slice).
- units as Scalar are problematic, because they are not closed under
  multiplication, i.e. m•m = m² != m. Can we deal with this in some
  way? They are only closed under addition...
- ???



Go with something like this for the general API:
```nim
type
  Tensor[T] = object
    d: seq[T]

proc len[T](t: Tensor[T]): int = t.d.len

proc `[]=`[T](t: var Tensor[T], idx: int, val: T) = t.d[idx] = val

proc init[T: Tensor](_: type T, num: int = 0): T =
  result = T(d: newSeq[float](num))

proc init[T: seq](_: type T, num: int = 0): T =
  result = newSeq[float](num)

let t = Tensor[float].init()
echo t

let s = seq[float].init()
echo s

proc linspace[T: Tensor | seq](t: type T, frm, to: float, steps: int): T =
  result = T.init(steps)
  let diff = (to - frm) / steps.float
  for i in 0 ..< result.len:
    result[i] = frm + i.float * diff

let t2 = Tensor[float].linspace(0.0, 10.0, 10)
echo t2
let s2 = seq[float].linspace(0.0, 10.0, 10)
echo s2
```
