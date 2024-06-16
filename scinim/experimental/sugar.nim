import std / [macros, math]
export math

## This module contains a whole bunch of fun little sugar procs / templates / macros
## to (mostly) help with writing math code. The most likely use case might be for
## illustrative / explanatory code that is close to math in nature already.


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

proc Σ*[T](s: openArray[T]): T = s.sum
proc Π*[T](s: openArray[T]): T = s.prod

proc √*[T](x: T): T = sqrt(x)
proc √*[T](x: openArray[T]): T = sqrt(x)

template Σ_i*(frm, to: int, body: untyped): untyped =
  var res: int
  for i {.inject.} in frm ..< to:
    res += body
  res

macro Σ_i*(col, body: untyped): untyped =
  let typ = getTypeReplaceI(body)
  let iId = ident"i"
  result = quote do:
    var res: typeof(`typ`)
    for `iId` in 0 ..< `col`.len:
      res += `body`
    res

macro λ*(arg, body: untyped): untyped =
  ##
  # XXX: Support multiple arguments!
  if arg.kind != nnkInfix or
     (arg.kind == nnkInfix and arg[0].kind in {nnkIdent, nnkSym} and arg[0].strVal != "->"):
    error("Unsupported operation in `λ`. The infix must be `->`, but is: " & arg[0].repr)
  let a = arg[1]
  let typ = arg[2]
  result = quote do:
    proc(`a`: `typ`): auto = `body`

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
  result = newProc(name = fnName, body = rhs)
  result[2] = genTyps
  result[3] = fnArgs

macro mathScope*(args: untyped): untyped =
  expectKind(args, nnkStmtList)
  result = newStmtList()
  for arg in args:
    result.add generateFunc(arg)
