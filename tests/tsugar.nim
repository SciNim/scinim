import ../scinim/experimental/sugar
import unittest

# Note: these are all highly experimental and subject to change!

suite "SciNim experimental sugar":
  test "Σ and Π short for `s.sum`, `s.prod`":
    let s1 = @[1,2,3]
    let s2 = @[1.0, 2.0, 3.0]
    check (Σ s1) == s1.sum
    check (Σ s2) == s2.sum
    check (Π s1) == s1.prod
    check (Π s2) == s2.prod

  test "Σ_i short for summing a collection with field access in a specific range":
    type
      Foo = object
        x: int
    let f = @[Foo(x: 1), Foo(x: 2), Foo(x: 3), Foo(x: 4)]

    # Long form taking start, stop indices and the body
    check Σ_i(0, f.len, f[i].x) == 10
    check Σ_i(1, f.len, f[i].x) == 9
    check Σ_i(2, f.len, f[i].x) == 7
    check Σ_i(3, f.len, f[i].x) == 4
    check Σ_i(2, f.len - 1, f[i].x) == 3

    # short form (similar to `Σ`), but allowing field acces etc
    # Injects the index `i`
    let res = Σ_i(f, f[i].x)
    check res == 10
    let res2 = Σ_i(f): f[i].x
    check res2 == 10
    check (block: # we're are a bit limited in what conditions the syntax parses correctly
             Σ_i(f): f[i].x
    ) == 10
    # i.e. this does not compile:
    # check (Σ_i(f): f[i].x) == 10
    check Σ_i(f, f[i].x) == 10

  test "λ to define 'lambdas', i.e. anonymous functions / closures":
    block:
      let fn = λ(x -> int): x*x
      check fn(2) == 4
      check fn(3) == 9
    block:
      let fn = λ(x -> int): x+x
      check fn(2) == 4
      check fn(3) == 6

  test "`mathScope` DSL that acts 'untyped'":
    mathScope:
      g(x) = exp(-x)
      h(x, μ, σ) = 1.0/sqrt(2*Pi) * exp(-pow(x - μ, 2) / (2 * σ*σ))

    check g(1.5) == exp(-1.5)
    check h(1.0, 0.5, 1.1) == 1.0/sqrt(2*Pi) * exp(-pow(1.0 - 0.5, 2) / (2 * 1.1^2))
