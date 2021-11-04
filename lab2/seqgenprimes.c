#ifndef __cplusplus
typedef unsigned char bool;
static const bool false = 0;
static const bool true = 1;
#endif
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define GEN_SEQ_RES
#define PRINT_PRIME_NUM

int main(int argc, char** argv)
{
    if(argc!=2)
        fprintf(stderr, "invalid parameter\n");
    // assume N>=1
    int N = atoi(argv[1]);
    bool* is_prime = (bool*)calloc(N+1,sizeof(bool));
    is_prime[0] = false;
    is_prime[1] = false;
    for(int i = 2; i<=N; ++i)
        is_prime[i] = true;
    int end_x = (int)ceil(sqrt(N));
    for(int x = 2; x <= end_x; ++x)
    {
        if(is_prime[x])
        {
            for(int y = x + 1; y <= N; ++ y)
            {
                if(y % x == 0)
                    is_prime[y] = false;
            }
        }
    }
    char out_path[20];    
    sprintf(out_path, "%d.txt", N);
    FILE* fp = fopen(out_path, "w");
#ifdef GEN_SEQ_RES
    char seq_out_path[20];
    sprintf(seq_out_path, "seq%d.txt", N);
    FILE* seq_fp = fopen(seq_out_path, "w");
#endif
#ifdef PRINT_PRIME_NUM
    int prime_num = 0;   
#endif
    for(int i = 1; i<=N; ++i)
    {
        if(is_prime[i])
        {
            fprintf(fp, "%d ", i);
#ifdef GEN_SEQ_RES
            fprintf(seq_fp, "%d ", i);
#endif
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
