#!/bin/bash

`clang++ -o xornn main.cpp -std=c++11 -lm`

```
$ ./xornn 
Input: 0 0 Output: 0.0168766
Input: 0 1 Output: 0.980629
Input: 1 0 Output: 0.980629
Input: 1 1 Output: 0.0184401
```