## sequential version
- gcc seqgenprimes.c -o seqgenprimes -lm -std=c99
- need -lm option to use math.h
- use a bool list to store if each number is a prime
- if not c++, define bool type manually

## cuda version
- nvcc -o genprimes genprimes.cu
- each block computes prime numbers in a range of BLOCK_SIZE
- each thread in a block uses corresponding divisors to find non-prime numbers
- after thread sync, summarize prime numbers of a block
- you can uncomment PRINT_PRIME_NUM macro to print nums