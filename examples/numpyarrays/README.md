# Example of calling Nim from Python

This examples shows how to call a Nim function from Python and how to use SciNim + Nimpy to loop directly over a Numpy array buffer.
When returning data be careful not to return dangling pointer accidently (hint: you'll see a segfault from Python)


## Running the example

To run the example you can simply compile nim and execute python : 

``nim c examply && python examply.py > results.txt``

All the nim compile-time options are in examply.nims (nothing fancy just the standard "fast" nim compile options).
To demonstrate how this speeds up Python execution loop, the examples has A LOT of element. Each Python loops takes about ~900 seconds and there is 3 of them so feel free to comment them if you are just interested in Nim's timing.
