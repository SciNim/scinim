import macros

proc traverseTree(input: NimNode): NimNode =
  # iterate children
  for i in 0 ..< input.len:
    case input[i].kind
    of nnkSym:
      # if we found a symbol, take it
      result = input[i]
    of nnkBracketExpr:
      # has more children, traverse
      result = traverseTree(input[i])
    else:
      error("Unsupported type: " & $input.kind)

macro getSubType*(TT: typed): untyped =
  ## macro to get the subtype of a nested type by iterating
  ## the AST
  # traverse the AST
  let res = traverseTree(TT.getTypeInst)
  # assign symbol to result
  result = quote do:
    `res`
  echo "GET SUB TYPE ", result.treerepr, " from ", TT.treerepr

macro getOuterType*(TT: typed): untyped =
  ## macro to get the subtype of a nested type by iterating
  ## the AST
  # traverse the AST
  let res = TT.getTypeInst
  doAssert res.kind == nnkBracketExpr
  doAssert res.typeKind == ntyTypeDesc
  let arg = res[1]
  doAssert arg.kind == nnkBracketExpr
  let outer = ident(arg[0].strVal)
  # assign symbol to result
  result = quote do:
    `outer`

proc stripTypedesc(n: NimNode): NimNode =
  case n.typeKind
  of ntyTypeDesc: result = n[1]
  else: result = n

macro outerType*(TT: typed, arg: typed): untyped =
  var res = TT.getTypeInst.stripTypedesc()
  echo "res ", res.treerepr
  echo TT.treerepr
  echo TT.typeKind
  #echo TT.getType.treerepr
  #echo TT.getTypeImpl.treerepr
  #echo TT.getTypeInst.treerepr
  if res.kind != nnkBracketExpr:
    res = TT.getType.stripTypedesc()
  #doAssert res.kind == nnkBracketExpr
  #if res.kind != nnkBracketExpr:
  #  echo "returning ", res.repr
  #  return nnkBracketExpr.newTree(res, arg)
  #  #return nnkNilLit.newNimNode() #nnkBracketExpr.newTree(ident"seq", ident"int")
  let outer = if res.len > 0: ident(res[0].strVal)
              else: res
  echo "OUTER ", outer.treerepr, " and arrg ", arg.treerepr
  result = nnkBracketExpr.newTree(outer, arg)
  echo "OUTER TYPE: ", result.treerepr

macro `->`*(call, typ: untyped): untyped =
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

template `<-`*(typ, call: untyped): untyped =
  call -> typ
