import std / [macros, options, algorithm]

type
  ForLoop = object
    n: NimNode # the actual node
    body: NimNode # body of the loop *WITHOUT* any inner loops!
    idx: NimNode # the loop index
    start: NimNode # start of the loop
    stop: NimNode # stop of the loop

template nofuse*(arg: untyped): untyped =
  ## Just a dummy template, which can be easily used to disable fusing of
  ## a nested loop
  arg

proc extractBody(n: NimNode): NimNode =
  ## Returns the input tree without any possible nested for loops. Nested
  ## loops are replaced by `nnkEmpty` nodes to be filled again later in `bodies`.
  case n.kind
  of nnkForStmt:
    if n[1].kind == nnkInfix and n[1][0].strVal == "..<":
      result = newEmptyNode() ## Flattened nested loop body will be inserted here
    else:
      result = n
  else:
    if n.len > 0:
      result = newTree(n.kind)
      for ch in n:
        let bd = extractBody(ch)
        if bd != nil:
          result.add bd
    else:
      result = n

proc toForLoop(n: NimNode): Option[ForLoop] =
  ## Returns a `some(ForLoop)` if the given node is a fuse-able for loop
  doAssert n.kind == nnkForStmt
  if n[1].kind != nnkInfix: return
  if n[1][0].strVal != "..<":
    error("Unexpected iterator: " & $n[1].repr &
      ". It must be of the form `0 ..< X`.")
  if not (n[1][1].kind == nnkIntLit and n[1][1].intVal == 0):
    error("Starting iteration index must be 0!")
  result = some(ForLoop(n: n,
                        body: extractBody(n[2]),
                        idx: n[0],
                        start: n[1][1],
                        stop: n[1][2]))

template addIf(s, opt): untyped =
  if opt.isSome:
    s.add opt.unsafeGet

proc extractLoops(n: NimNode): seq[ForLoop] =
  ## Extracts (fuse-able) loops from the given Nim node and errors if more than
  ## one for loop found at the same level.
  case n.kind
  of nnkForStmt:
    result.addIf toForLoop(n)
    result.add extractLoops(n[2]) # go over body
  else:
    var foundLoops = 0 # counter for number of loops at current body
    for ch in n:
      let loops = extractLoops(ch)
      if loops.len > 0:
        result.add loops
        inc foundLoops
    if foundLoops > 1:
      error("Found more than one loop (" & $foundLoops & ") at the level of node: " &
        n.repr & ". Please wrap " & "these loops as `nofuse`, i.e. `nofuse(0 ..< X)`")

proc genFusedLoop(idx: NimNode, stop: NimNode, ompStr = ""): NimNode =
  ## Generate either regular or OpenMP for loop
  let loopIter = if ompStr.len == 0:
                   nnkInfix.newTree(ident"..<",
                                    newLit 0,
                                    stop)
                 else:
                   nnkCall.newTree(ident"||",
                                   newLit 0,
                                   stop,
                                   newLit ompStr)
  result = nnkForStmt.newTree(
    idx,
    loopIter
  )

proc calcStop(loops: seq[ForLoop]): NimNode =
  ## Returns `N * T * U * ...` expression where the indices are
  ## the stop indices of the loops to be fused.
  case loops.len
  of 0: doAssert false, "Must not happen"
  of 1: result = loops[0].stop
  else:
    var ml = loops.reversed # want last elements first
    let x = ml.pop
    result = nnkInfix.newTree(ident"*", x.stop,
                              calcStop(ml.reversed))

proc modOrDiv(prefix, suffix: NimNode, isDiv: bool): NimNode =
  if isDiv:
    result = quote do:
      `prefix` div `suffix`
  else:
    result = quote do:
      `prefix` mod `suffix`

proc asLet(v, val: NimNode): NimNode =
  result = quote do:
    let `v` = `val`

proc genPrelude(idx: NimNode, loops: seq[ForLoop]): NimNode =
  ## The basic algorithm for generating the correct index for fused loops is
  ##
  ## Notation:
  ## `i` = Loop index of single remaining outer loop
  ## `N_i` = Stopping index (-1) of the inner loop `i`
  ## `n` = Total number of nested loops
  ##
  ## Whichever is easiest to read for you:
  ##
  ## `let i0 = i div (N_0 * N_1 ... N_n)`
  ## `let i1 = (i mod (N_0 * N_1 ... N_n)) div (N_1 * N_2 * ... N_n)`
  ## `let i2 = ((i mod (N_0 * N_1 ... N_n)) mod (N_1 * N_2 * ... N_n)) div (N_2 * ... * N_n)`
  ## ...
  ##
  ## ... or
  ##
  ## `let i0 = i div Π_i=0^n N_i`
  ## `let i1 = (i mod Π_i=0^n N_i) div Π_i=1^n N_i`
  ## `let i2 = ((i mod Π_i=0^n N_i) mod Π_i=1^n N_i) div Π_i=2^n N_i`
  ##
  ## ...or
  ##
  ## `let i0 = Idx div [Product of remaining N-1 loops]`
  ## `let i1 = (Idx mod [Product of remaining loops]) div [Product of remaining N-2 loops]`
  ## `let i2 = (Idx mod [Product of remaining loops]) mod [Product of remaining N-2 loops]`
  result = newStmtList()
  var prefix = idx
  var ml = loops.reversed
  var lIdx = ml.pop # drop first element
  var suffix = ml.calcStop()
  while ml.len > 0:
    result.add asLet(lIdx.idx, modOrDiv(prefix, suffix, isDiv = true))
    lIdx = ml.pop # get next loop index & adjust remaining loops
    # now adjust prefix and suffix
    prefix = modOrDiv(prefix, suffix, isDiv = false)
    if ml.len > 0: # adjust suffix
      suffix = ml.calcStop()
    else: # simply add last 'prefix'
      result.add asLet(lIdx.idx, prefix)

proc bodies(loops: seq[ForLoop]): NimNode =
  ## Concatenates all loop bodies, by placing the next loop into the
  ## `nnkEmpty` node of the current node
  var ml = loops.reversed
  #echo ml.repr
  var cur = ml.pop
  result = cur.body
  for i in 0 ..< result.len:
    let ch = result[i]
    if ch.kind == nnkEmpty:
      # insert next loop
      result[i] = bodies(ml.reversed) # revert order again
      break # there can only be a single `nnkEmpty` (multiple loops not allowed,
            # yields CT error)

proc fuseLoopImpl(ompStr: string, body: NimNode): NimNode =
  # 1. extract all loops from the body
  let loops = extractLoops(body)
  # 2. generate identifier for the final loop
  let idx = genSym(nskForVar, "idx")
  # 3. generate the fused outer loop
  result = genFusedLoop(idx, calcStop(loops), ompStr)
  # 4. generate final loop body by...
  var loopBody = newStmtList()
  # 4a. generate prelude of loop variables of original loops
  loopBody.add genPrelude(idx, loops) # gen code to produce the old loop variables
  # 4b. insert old loop bodies into respective positions
  loopBody.add bodies(loops)
  result.add loopBody
  when defined(DebugFuseLoop):
    echo result.repr

macro fuseLoops*(body: untyped): untyped =
  result = fuseLoopImpl("", body)

macro fuseLoops*(ompStr: untyped{lit}, body: untyped): untyped =
  result = fuseLoopImpl(ompStr.strVal, body)
