# Package
version       = "0.2.4"
author        = "SciNim"
description   = "The core types and functions of the SciNim ecosystem"
license       = "MIT"

# C++ codegen catches more bugs than C
backend       = "cpp"

# Dependencies
requires "nim >= 1.4.0"
requires "fusion"
requires "arraymancer >= 0.7.3"
requires "polynumeric >= 0.2.0"
requires "nimpy >= 0.2.0"
