import math
proc Σ*[T](s: openArray[T]): T = s.sum
proc Π*[T](s: openArray[T]): T = s.prod

let s = @[1,2,3]
echo Σ s
echo Π s

template Σ_i*(frm, to: int, body: untyped): untyped =
  var res: int
  for i {.inject.} in frm ..< to:
    res += body
  res

#template Σ_i(to: int, body: untyped): untyped =
#  #var res: typeof(s[0])
#  var res: int
#  for i {.inject.} in 0 ..< to:
#    res += body
#  res

import macros
proc getTypeReplaceI(arg: NimNode): NimNode =
  if arg.len > 0:
    result = newTree(arg.kind)
    for ch in arg:
      result.add getTypeReplaceI(ch)
  else:
    case arg.kind
    of nnkIdent, nnkSym:
      if arg.strVal == "i": return newLit(0)
      else: return arg
    else: return arg


macro Σ_i*(col, body: untyped): untyped =
  let typ = getTypeReplaceI(body)
  let iId = ident"i"
  result = quote do:
    var res: typeof(`typ`)
    for `iId` in 0 ..< `col`.len:
      res += `body`
    res
  echo result.repr

type
  Foo = object
    x: int
let f = @[Foo(x: 1), Foo(x: 4)]

echo Σ_i(0, f.len, f[i].x)
let r = Σ_i(f): f[i].x
echo r

macro λ*(arg, body: untyped): untyped =
  let a = arg[1]
  let typ = arg[2]
  echo body.repr
  result = quote do:
    let fn = proc(`a`: `typ`): auto = `body`
    fn

let fn = λ(x -> int): x*x
echo fn(2)

proc sliceTypes(n: NimNode, sl: Slice[int]): tuple[args, genTyps: NimNode] =
  var args = nnkFormalParams.newTree(ident"auto")
  var genTyps = nnkIdentDefs.newTree()
  for i in sl.a .. sl.b:
    let typ = ident($char('A'.ord + i - 1))
    args.add nnkIdentDefs.newTree(n[i],
                                  typ,
                                  newEmptyNode())
    genTyps.add typ
  genTyps.add newEmptyNode()
  genTyps.add newEmptyNode()
  genTyps = nnkGenericParams.newTree(genTyps)
  result = (args: args, genTyps: genTyps)

proc generateFunc(arg: NimNode): NimNode =
  expectKind(arg, nnkAsgn)
  let lhs = arg[0]
  let rhs = arg[1]
  let fnName = lhs[0]
  let (fnArgs, genTyps) = sliceTypes(lhs, 1 ..< lhs.len)
  echo fnArgs.treerepr
  result = newProc(name = fnName, body = rhs)
  result[2] = genTyps
  result[3] = fnArgs
  echo result.repr

macro mathScope*(args: untyped): untyped =
  echo args.treerepr
  expectKind(args, nnkStmtList)
  result = newStmtList()
  for arg in args:
    result.add generateFunc(arg)
  echo result.repr

mathScope:
  g(x) = exp(-x)
  h(x, μ, σ) = 1.0/sqrt(2*Pi) * exp(-pow(x - μ, 2) / (2 * σ*σ))

echo g(1.5)
echo h(1.0, 0.5, 1.1)
