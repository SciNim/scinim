import macro_utils
#import arraymancer / tensor
import arraymancer/laser/strided_iteration/foreach

import arraymancer / tensor
export Tensor, rank, size, `[]`, `[]=`, toTensor, unsafe_raw_offset, is_C_contiguous


func len*[T](t: Tensor[T]): int = t.size

iterator items*[T](t: Tensor[T]): T =
  doAssert t.rank == 1
  when T is KnownSupportsCopyMem:
    forEach x in t:
      yield x
  else:
    for i in 0 ..< t.len:
      yield t[i]

iterator pairs*[T](t: Tensor[T]): (int, T) =
  doAssert t.rank == 1
  when T is KnownSupportsCopyMem:
    var idx = 0
    forEach x in t:
      yield (idx, x)
      inc idx
  else:
    for i in 0 ..< t.len:
      yield (i, t[i])

## newLike needs this weird signature, because otherwise we get a `newLike` leads to
## an ambiguous overload error
proc newLike*[T: Tensor](dtype: typedesc[T], size: int): T =
  result = newTensor[getSubType(T)](size)
