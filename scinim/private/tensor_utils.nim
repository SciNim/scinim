import macro_utils
#import arraymancer / tensor
import arraymancer/laser/strided_iteration/foreach

import arraymancer / tensor
export Tensor, rank, size, `[]`, `[]=`, toTensor, unsafe_raw_offset, is_C_contiguous# , display, `$`

func len*[T](t: Tensor[T]): int = t.size

iterator iter*[T](t: Tensor[T]): T =
  doAssert t.rank == 1
  when T is KnownSupportsCopyMem:
    forEach x in t:
      yield x
  else:
    for i in 0 ..< t.len:
      yield t[i]

iterator miter*[T](t: var Tensor[T]): var T =
  doAssert t.rank == 1
  when T is KnownSupportsCopyMem:
    forEach x in t:
      yield x
  else:
    for i in 0 ..< t.len:
      yield t[i]

iterator pairIter*[T](t: Tensor[T]): (int, T) =
  doAssert t.rank == 1
  when T is KnownSupportsCopyMem:
    var idx = 0
    forEach x in t:
      yield (idx, x)
      inc idx
  else:
    for i in 0 ..< t.len:
      yield (i, t[i])

## init needs this weird signature, because otherwise we get a `init` leads to
## an ambiguous overload error
proc init*[T: Tensor](_: typedesc[T], size: int = 0): T =
  result = newTensor[getSubType(T)](size)

proc init*[T: Tensor; U](_: typedesc[T], inner: typedesc[U], size: int = 0): Tensor[T] =
  result = Tensor[U].init()
