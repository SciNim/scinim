# scinim
The core types and functions of the SciNim ecosystem


## Vector / Scalar features we need

- sorting: can this work properly? At least if we allow copy, but
  without copy we are in trouble with views (e.g. a tensor slice).
- units as Scalar are problematic, because they are not closed under
  multiplication, i.e. m•m = m² != m. Can we deal with this in some
  way? They are only closed under addition...
- ???
