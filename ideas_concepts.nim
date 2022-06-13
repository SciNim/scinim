type
  Tensor[T] = object
    d: seq[T]

  Seq = object

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

import macros
macro `->`(call, typ: untyped): untyped =
  #echo call.treerepr
  #echo typ.treerepr
  var nameN = call[0]
  var innerTyp: NimNode
  case nameN.kind
  of nnkBracketExpr:
    innerTyp = nameN[1]
    nameN = nameN[0]
  else: innerTyp = ident"float"
  var nCall = nnkCall.newTree(nameN)
  if typ.strVal == "Tensor":
    nCall.add nnkBracketExpr.newTree(ident"Tensor", innerTyp)
  elif typ.strVal == "seq":
    nCall.add nnkBracketExpr.newTree(ident"seq", innerTyp)
  for i in 1 ..< call.len:
    let ch = call[i]
    nCall.add ch
  result = nCall
  echo result.treerepr

template `<-`(typ, call: untyped): untyped =
  call -> typ

echo linspace(0.0, 10.0, 10) -> Tensor
echo linspace[float](0.0, 10.0, 10) -> Tensor
echo linspace(0.0, 10.0, 10) -> seq

echo Tensor <- linspace(0.0, 10.0, 10)
echo seq <- linspace(0.0, 10.0, 10)


proc init[T: seq; U](_: typedesc[T], inner: typedesc[U], size: int = 0): seq[U] =
  result = seq[U].init(size)

#echo seq.init(10)
