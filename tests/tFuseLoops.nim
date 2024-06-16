import ../scinim/fuse_loops
import std / unittest

suite "fuseLoops":
  test "Compiles test for different `fuseLoops` setups":
    const N = 5
    const T = 10
    const X = 3

    ## XXX: These should probably become proper tests. :)

    fuseLoops:
      for i in 0 ..< N:
        let x = i * 2
        for j in 0 ..< T:
          let z = x * j
          echo i, j, x, z
        echo x

    fuseLoops:
      for i in 0 ..< N:
        let x = i * 2
        for j in 0 ..< T:
          let z = x * j
          echo i, j, x, z
          for k in nofuse(0 ..< T):
            echo k
        echo x

    fuseLoops("parallel for"):
      for i in 0 ..< N:
        let x = i * 2
        for j in 0 ..< T:
          let z = x * j
          for k in 0 ..< X:
            echo i, j, k, x, z
        echo x

    ## The following raises a CT error
    when compiles((
      fuseLoops:
        for i in 0 ..< N:
          let x = i * 2
          var zsum = 0
          for j in 0 ..< T:
            let z = x * j
            zsum += z
            echo i, x, z
          echo x
          for j in 0 ..< 2 * T:
            zsum += j
          echo zsum
    )):
      doAssert false
