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
requires "arraymancer >= 0.7.32"
requires "polynumeric >= 0.2.0"
requires "nimpy >= 0.2.0"
requires "print"

task test, "Run all tests":
  echo "Running tests command"
  exec "nim cpp -r --mm:Atomicarc tests/tnumpyarrays.nim"
  exec "nim cpp -r --mm:orc tests/tnumpyarrays.nim"
