#ifndef __cplusplus
typedef unsigned char bool;
static const bool false = 0;
static const bool true = 1;
#endif
#include <cuda.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #define PRINT_PRIME_NUM
#define BLOCK_SIZE 100

__global__ void gen_prime_kernel(bool* _is_prime, int _max_divisor)
{
    // start from blockid*blocksize, size is blocksize
    __shared__ int not_prime[BLOCK_SIZE][BLOCK_SIZE];
    int begin_x = blockIdx.x * BLOCK_SIZE + 1; // included
    for(int i = 0; i < BLOCK_SIZE; ++ i)
        not_prime[threadIdx.x][i] = 0;
    for(int divisor = int(threadIdx.x); divisor <= _max_divisor; divisor+=BLOCK_SIZE)
    {
        if(divisor == 0 || divisor == 1)
            continue;
        for(int i = 0; i < BLOCK_SIZE; ++ i)
        {
            if((begin_x + i) % divisor == 0 && begin_x + i > divisor)
                not_prime[threadIdx.x][i] = 1;
        }
    }
    __syncthreads();
    int thread_not_prime = 0;
    for(int i = 0; i < BLOCK_SIZE; ++ i)
        thread_not_prime += not_prime[i][threadIdx.x];
    bool thread_prime = !thread_not_prime;
    _is_prime[begin_x + threadIdx.x] = thread_prime;
    // _is_prime[begin_x + threadIdx.x] = true;
}

int main(int argc, char** argv)
{
    if(argc!=2)
        fprintf(stderr, "invalid parameter\n");
    // assume N>=1
    int N = atoi(argv[1]);
    bool* is_prime = (bool*)calloc(N+1,sizeof(bool));
    int max_divisor = (int)ceil(sqrt(N));
    bool* d_is_prime;
    cudaMalloc((void**)&d_is_prime, (N + BLOCK_SIZE + 1)*sizeof(bool));
    gen_prime_kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_is_prime, max_divisor);
    cudaMemcpy(is_prime, d_is_prime, (N+1)*sizeof(bool), cudaMemcpyDeviceToHost);
    is_prime[0] = false;
    is_prime[1] = false;
    char out_path[20];    
    sprintf(out_path, "%d.txt", N);
    FILE* fp = fopen(out_path, "w");

#ifdef PRINT_PRIME_NUM
    int prime_num = 0;   
#endif
    for(int i = 1; i<=N; ++i)
    {
        if(is_prime[i])
        {
            fprintf(fp, "%d ", i);

#ifdef PRINT_PRIME_NUM
            ++prime_num;
#endif
        }
    }
#ifdef PRINT_PRIME_NUM
    printf("prime num: %d\n", prime_num);   
#endif 
    free(is_prime);
    return 0;
}
