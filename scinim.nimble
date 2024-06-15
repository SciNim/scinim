# Package
version       = "0.2.5"
author        = "SciNim"
description   = "The core types and functions of the SciNim ecosystem"
license       = "MIT"

# C++ codegen catches more bugs than C
backend       = "cpp"

# Dependencies
requires "nim >= 1.6.0"
requires "threading"
requires "arraymancer >= 0.7.31"
requires "polynumeric >= 0.2.0"
requires "nimpy >= 0.2.0"

task test, "Run all tests":
  exec "nim c -r tests/tnumpyarrays.nim"
  exec "nim c -r --gc:orc tests/tnumpyarrays.nim"
  exec "nim cpp -r tests/tnumpyarrays.nim"
  exec "nim cpp -r --gc:orc tests/tnumpyarrays.nim"
