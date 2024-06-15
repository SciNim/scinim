## Module that implements several procedures related to prime numbers
##
## Prime numbers are an essential building block of many algorithms in diverse
## areas such as cryptography, digital communications and many others.
## This module adds a function to generate rank-1 tensors of primes upto a
## certain value; as well as a function to calculate the prime factors of a
## number.

import arraymancer

proc primes*[T: SomeInteger | SomeFloat](upto: T): Tensor[T] =
  ## Generate a Tensor of prime numbers up to a certain value
  ##
  ## Return a Tensor of the prime numbers less than or equal to `upto`.
  ## A prime number is one that has no factors other than 1 and itself.
  ##
  ## Input:
  ##   - upto: Integer up to which primes will be generated
  ##
  ## Result:
  ##   - Integer Tensor of prime values less than or equal to `upto`
  ##
  ## Note:
  ##   - This function implements a "half" Sieve of Erathostenes algorithm
  ##     which is a classical Sieve of Erathostenes in which only odd numbers
  ##     are checked. Many examples of this algorithm can be found online.
  ##     It also stops checking after sqrt(upto)
  ##   - The memory required by this procedure is proportional to the input
  ##     number.
  when T is SomeFloat:
    if upto != round(upto):
      raise newException(ValueError,
        "`upto` value (" & $upto & ") must be a whole number")

  if upto < 11:
    # Handle the primes below 11 to simplify the general code below
    # (by removing the need to handle the few cases in which the index to
    # `isprime`, calculated based on `factor` is negative)
    # This is the minimum set of primes that we must handle, but we could
    # extend this list to make the calculation faster for more of the
    # smallest primes
    let prime_candidates = [2.T, 3, 5, 7].toTensor()
    return prime_candidates[prime_candidates <=. upto]

  # General algorithm (valid for numbers higher than 10)
  let prime_candidates = arange(3.T, T(upto + 1), 2.T)
  var isprime = ones[bool]((upto.int - 1) div 2)
  let max_possible_factor_idx = int(sqrt(upto.float)) div 2
  for factor in prime_candidates[_ ..< max_possible_factor_idx]:
    if isprime[(factor.int - 2) div 2]:
      isprime[(factor.int * 3 - 2) div 2 .. _ | factor.int] = false

  # Note that 2 is missing from the result, so it must be manually added to
  # the front of the result tensor
  return [2.T].toTensor().append(prime_candidates[isprime])

# The maximum float64 that can be represented as an integer that is followed by a
# another integer that is representable as a float64 as well
const maximumConsecutiveFloat64Int = pow(2.0, 53) - 1.0

proc factor*[T: SomeInteger | SomeFloat](n: T): Tensor[T] =
  ## Return a Tensor containing the prime factors of the input
  ##
  ## Input:
  ##   - n: A value that will be factorized.
  ##        If its type is floating-point it must be a whole number. Otherwise
  ##        a ValueError will be raised.
  ## Result:
  ##   - A sorted Tensor containing the prime factors of the input.
  ##
  ## Example:
  ## ```nim
  ## echo factor(60)
  ## # Tensor[system.int] of shape "[4]" on backend "Cpu"
  ## #     2     2     3     5
  ## ```
  if n < 0:
    raise newException(ValueError,
      "Negative values (" & $n & ") cannot be factorized")
  when T is int64:
    if n > T(maximumConsecutiveFloat64Int):
      raise newException(ValueError,
        "Value (" & $n & ") is too large to be factorized")
  elif T is SomeFloat:
    if floor(n) != n:
      raise newException(ValueError,
        "Non whole numbers (" & $n & ") cannot be factorized")

  if n < 4:
    return [n].toTensor

  # The algorithm works by keeping track of the list of unique potential,
  # candidate prime factors of the input, and iteratively adding those
  # that are confirmed to be factors into a list of confirmed factors
  # (which is stored in the `result` tensor variable).

  # First we must initialize the `candidate_factor` Tensor
  # The factors of the input can be found among the list of primes
  # that are smaller than or equal to input. However we can significantly
  # reduce the candidate list by taking into account the fact that only a
  # single factor can be greater than the square root of the input.
  # The algorithm is such that if that is the case we will add the input number
  # at the very end of the loop below
  var candidate_factors = primes(T(ceil(sqrt(float(n)))))

  # This list of prime candidate_factors is refined by finding those of them
  # that divide the input value (i.e. those whose `input mod prime` == 0).
  # Those candiates that don't divide the input are known to not be valid
  # factors and can be removed from the candidate_factors list. Those that do
  # divide the input are confirmed as valid factors and as such are added to
  # the result list. Then the input is divided by all of the remaining
  # candidates (by dividing the input by the product of all the remaining
  # candidates). The result is a number that is the product of all the factors
  # that are still unknown (which must be among the remaining candidates!) and
  # which we can call `unknown_factor_product`.
  # Then we can simply repeat the same process over and over, replacing the
  # original input with the remaining `unknown_factor_product` after each
  # iteration, until the `unknown_factors_product` (which is reduced by each
  # division at the end of each iteration) reaches 1. Alternatively, we might
  # run out of candidates, which will only happen when there is only one factor
  # left (which must be greater than the square root of the input) and is stored
  # in the `unknown_factors_product`. In that case we add it to the confirmed
  # factors (result) list and the process can stop.
  var unknown_factor_product = n
  while unknown_factor_product > 1:
    # Find the primes that are divisors of the remaining unknown_factors_product
    # Note that this tells us which of the remaining candidate_factors are
    # factors of the input _at least once_ (but they could divide it more
    # than once)
    let is_factor = (unknown_factor_product mod candidate_factors) ==. 0
    # Keep only the factors that divide the remainder and remove the rest
    # from the list of candidates
    candidate_factors = candidate_factors[is_factor]
    # after this step, all the items incandidate_factors are _known_ to be
    # factors of `unknown_factor_product` _at least once_!
    if candidate_factors.len == 0:
      # If there are no more prime candidates left, it means that the remainder
      # is a prime (and that it must be greater than the sqrt of the input),
      # and that we are done (after adding it to the result list)
      result = result.append([unknown_factor_product].toTensor)
      break
    # If we didn't stop it means that there are still candidates which we
    # _know_ are factors of the remainder, so we must add them to the result
    result = result.append(candidate_factors)
    # Now we can prepare for the next iteration by dividing the remainder,
    # by the factors we just added to the result. This reminder is the product
    # of the factors we don't know yet
    unknown_factor_product = T(unknown_factor_product / product(candidate_factors))
    # After this division the items in `candidate_factors` become candidates again
    # and we can start a new iteration
  result = sorted(result)

proc isprimeImpl[T: SomeInteger | SomeFloat](n: T, candidate_factors: Tensor[int]): bool {.inline.} =
  ## Actual implementation of the isprime check
  # This function is optimized for speed in 2 ways:
  # 1. By first rejecting all non-whole float numbers and then performing the
  #    actual isprime check using integers (which is faster than using floats)
  # 2. By receving a pre-calculated tensor of candidate_factors. This does not
  #    speed up the check of a single value, but it does speed up the check of
  #    a tensor of values. Note that because of #1 the candidate_factors must
  #    be a tensor of ints (even if the number being checked is a float)
  when T is SomeFloat:
    if floor(n) != n:
      return false
    let n = int(n)
  result = (n > 1) and all(n mod candidate_factors[candidate_factors <. n])

proc isprime*[T: SomeInteger | SomeFloat](n: T): bool =
  ## Check whether the input is a prime number
  ##
  ## Only positive values higher than 1 can be primes (i.e. we exclude 1 and -1
  ## which are sometimes considered primes).
  ##
  ## Note that this function also works with floats, which are considered
  ## non-prime when they are not whole numbers.
  # Note that we do here some checks that are repeated later inside of
  # `isprimeImpl`. This is done to avoid the unnecessary calculation of
  # the `candidate_factors` tensor in those cases
  if n <= 1:
    return false
  when T is int64:
    if n > T(maximumConsecutiveFloat64Int):
      raise newException(ValueError,
        "Value (" & $n & ") is too large to be factorized")
  elif T is SomeFloat:
    if floor(n) != n:
      return false
  var candidate_factors = primes(int(ceil(sqrt(float(n)))))
  isprimeImpl(n, candidate_factors)

proc isprime*[T: SomeInteger | SomeFloat](t: Tensor[T]): Tensor[bool] =
  ## Element-wise check if the input values are prime numbers
  result = zeros[bool](t.len)
  # Pre-calculate the list of primes that will be used for the element-wise
  # isprime check and then call isprimeImpl on each element
  # Note that the candidate_factors must be a tensor of ints (for performance
  # reasons)
  var candidate_factors = primes(int(ceil(sqrt(float(max(t.flatten()))))))
  for idx, val in t.enumerate():
    result[idx] = isprimeImpl(val, candidate_factors)
  return result.reshape(t.shape)
