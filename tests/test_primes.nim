import ../scinim/primes
import arraymancer
import std / [unittest, random]

proc test_primes() =
  ## Test the `primes` function
  test "Prime number generation (integer values)":
    check: primes(0).len == 0
    check: primes(1).len == 0
    check: primes(2) == [2].toTensor
    check: primes(3) == [2, 3].toTensor
    check: primes(4) == [2, 3].toTensor
    check: primes(11) == [2, 3, 5, 7, 11].toTensor
    check: primes(12) == [2, 3, 5, 7, 11].toTensor
    check: primes(19) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(20) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(22) == [2, 3, 5, 7, 11, 13, 17, 19].toTensor
    check: primes(100000).len == 9592
    check: primes(100003).len == 9593
    check: primes(100000)[^1].item == 99991

  test "Prime number generation (floating-point values)":
    check: primes(100000.0).len == 9592
    check: primes(100000.0)[^1].item == 99991.0

    # An exception must be raised if the `upto` value is not a whole number
    try:
      discard primes(100.5)
      check: false
    except ValueError:
      # This is what should happen!
      discard

proc generate_random_factor_tensor[T](
    max_value: T, max_factors: int, prime_list: Tensor[T]): Tensor[T] =
  ## Generate a tensor of random prime factors taken from a tensor of primes
  ## The tensor length will not exceed the `max_factors` and the product of
  ## the factors will not exceed `max_value` either.
  ## This is not just a random list of values taken from the `prime_list`
  ## Instead we artificially introduce a random level of repetition of the
  ## chosen factors to emulate the fact that many numbers have repeated
  ## prime factors
  let max_value = rand(4 .. 2 ^ 53)
  let max_factors = rand(1 .. 20)
  result = newTensor[int](max_factors)
  var value = 1
  var factor = prime_list[rand(prime_list.len - 1)]
  for idx in 0 ..< max_factors:
    # Randomly repeat the previous factor
    # Higher number of repetitions are less likely
    let repeat_factor = rand(5) < 1
    if not repeat_factor:
      factor = prime_list[rand(prime_list.len - 1)]
    let new_value = factor * value
    if new_value >= max_value:
      break
    result[idx] = factor
    value = new_value
  result = sorted(result)
  result = result[result >. 0]

proc test_factor() =
  test "Prime factorization of integer values (factor)":
    check: factor(60) == [2, 2, 3, 5].toTensor
    check: factor(100001) == [11, 9091].toTensor

    # Check that the product of the factorization of a few random values
    # equals the original numbers
    for n in 0 ..< 10:
      let value = rand(10000)
      check: value == product(factor(value))

    # Repeat the previous test in a more sophisticated manner
    # Instead of generating random values and checking that the
    # product of their factorization is the same as the original values
    # (which could work for many incorrect implementations of factor),
    # generate a few random factor tensors, multiply them to get
    # the number that has them as prime factors, factorize those numbers
    # and check that their factorizations matches the original tensors
    let prime_list = primes(100)
    for n in 0 ..< 10:
      let max_value = rand(4 .. 2 ^ 53)
      let max_factors = rand(1 .. 20)
      var factors = generate_random_factor_tensor(
        max_value, max_factors, prime_list)
      let value = product(factors)
      check: factor(value) == factors

  test "Prime factorization of floating-point values (factor)":
    # Floating-point
    check: factor(60.0) == [2.0, 2, 3, 5].toTensor
    check: factor(100001.0) == [11.0, 9091].toTensor

    # Check that the product of the factorization of a few random values
    # equals the original numbers
    # Note that here we do not also do the reverse check (as we do for ints)
    # in order to make the test faster
    for n in 0 ..< 10:
      let value = floor(rand(10000.0))
      check: value == product(factor(value))

    # An exception must be raised if we try to factorize a non-whole number
    try:
      discard factor(100.5)
      check: false
    except ValueError:
      # This is what should happen!
      discard

proc test_isprime() =
  test "isprime":
    check: isprime(7) == true
    check: isprime(7.0) == true
    check: isprime(7.5) == false
    check: isprime(1) == false
    check: isprime(0) == false
    check: isprime(-1) == false
    let t = [
      [-1, 0, 2,  4],
      [ 5, 6, 7, 11]
    ].toTensor
    let expected = [
      [false, false, true, false],
      [ true, false, true,  true]
    ].toTensor
    check: isprime(t) == expected
    check: isprime(t.asType(float)) == expected

# Run the tests
suite "Primes":
  test_primes()
  test_factor()
  test_isprime()
