#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#define BLOCK_SIZE 10

__global__ void init_kernel(int _n, double* _unknown_arr_d_new);

// most of the calculation, a(i)(j)*x
__global__ void calc_kernel(int _n, int _block_num_per_unknown, double* _a_arr_d, double* _unknown_arr_d_prev, double* _unknown_arr_d_new);

// final caculation for unknowns, calc err arr, copy new to old unknowns, reset new unknowns to 0
__global__ void postprocess_kernel(int _n, double* _c_arr_d, double* _diag_arr_d, double* _unknown_arr_d_prev, double* _unknown_arr_d_new, \
double* _err_d);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "should have 1 parameter\n");
        exit(0);
    }
    std::ifstream ifs(argv[1]);
    int unknown_num, max_err;
    ifs >> unknown_num;
    ifs >> max_err;
    double* unknown_arr = new double[unknown_num];
    double* a_arr = new double[unknown_num*unknown_num]; // matrix A
    double* diag_arr = new double[unknown_num];
    double* c_arr = new double[unknown_num];// vector C
    double* rel_err = new double;
    for(int i = 0; i < unknown_num; ++ i)
    {
        ifs >> unknown_arr[i];
    }
    for(int i = 0; i < unknown_num; ++ i)
    {
        for(int j = 0; j < unknown_num; ++ j)
            ifs >> a_arr[i* unknown_num + j];
        ifs >> c_arr[i];
    }
    ifs.close();
    // coefficients of rewriting
    for(int i = 0; i < unknown_num; ++ i)
        diag_arr[i] = a_arr[i*unknown_num + i];
    for(int i = 0; i < unknown_num; ++ i)
        for(int j = 0; j < unknown_num; ++ j)
        {
            if(i == j)
                a_arr[i*unknown_num+i] = 0;
            else
                a_arr[i*unknown_num+j] = -a_arr[i*unknown_num+j];
        }
    double *unknown_arr_d_prev, *unknown_arr_d_new, *a_arr_d, *c_arr_d, *diag_arr_d, *err_d;
    int vec_bytes = unknown_num * sizeof(double);
    int mat_bytes = unknown_num * unknown_num * sizeof(double);
    cudaMalloc((void**)&unknown_arr_d_prev, vec_bytes);
    cudaMalloc((void**)&unknown_arr_d_new, vec_bytes);
    cudaMalloc((void**)&a_arr_d, mat_bytes);
    cudaMalloc((void**)&c_arr_d, vec_bytes);
    cudaMalloc((void**)&diag_arr_d, vec_bytes);
    cudaMalloc((void**)&err_d, sizeof(double));
    cudaMemcpy(unknown_arr_d_prev, unknown_arr, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(a_arr_d, a_arr, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_arr_d, c_arr, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(diag_arr_d, diag_arr, vec_bytes, cudaMemcpyHostToDevice);
    init_kernel<<<unknown_num / BLOCK_SIZE + 1, BLOCK_SIZE>>>(unknown_num, unknown_arr_d_new);
    int block_num_per_unknown = unknown_num / BLOCK_SIZE + 1;
    int block_num = block_num_per_unknown * unknown_num;
    int iter_num = 0;
    while(true)
    {
        ++ iter_num;
        calc_kernel<<< block_num, BLOCK_SIZE>>>(unknown_num, block_num_per_unknown, a_arr_d, \
        unknown_arr_d_prev, unknown_arr_d_new);
        postprocess_kernel<<< unknown_num / BLOCK_SIZE + 1, BLOCK_SIZE>>>(unknown_num, c_arr_d, diag_arr_d, \
        unknown_arr_d_prev, unknown_arr_d_new, err_d);
        cudaMemcpy(rel_err, err_d, sizeof(double), cudaMemcpyDeviceToHost);
        if((*rel_err) <= max_err)
            break;
    }
    cudaMemcpy(unknown_arr, unknown_arr_d_prev, vec_bytes, cudaMemcpyDeviceToHost);
    char outfile_name[100];
    sprintf(outfile_name, "%d.sol", unknown_num);
    std::ofstream ofs(outfile_name);
    for(int i = 0; i < unknown_num; ++i)
    {
        ofs<<unknown_arr[i]<<std::endl;
    }
    ofs.close();
    printf("total number of iterations: %d\n", iter_num);
    delete[] unknown_arr;
    delete[] a_arr;
    delete[] diag_arr;
    delete[] c_arr;
    delete rel_err;
    return 0;
}

__global__ void init_kernel(int _n, double* _unknown_arr_d_new)
{
    int unknown_i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(unknown_i < _n)
    {
        _unknown_arr_d_new[unknown_i] = 0;
    }
}

__global__ void calc_kernel(int _n, int _block_num_per_unknown, double* _a_arr_d, \
double* _unknown_arr_d_prev, double* _unknown_arr_d_new)
{
    int unknown_i = blockIdx.x / _block_num_per_unknown;
    __shared__ int x; // part of new x[i]
    if(threadIdx.x == 0)
        x = 0;
    if(unknown_i < _n)
    {
        int j = (blockIdx.x % _block_num_per_unknown) * BLOCK_SIZE + threadIdx.x;
        if(j < _n)
        {
            atomicAdd(&x, _a_arr_d[unknown_i * _n + j] * _unknown_arr_d_prev[j]);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
    {
        atomicAdd(&(_unknown_arr_d_new[unknown_i]), x);
    }
}

__global__ void postprocess_kernel(int _n, double* _c_arr_d, double* _diag_arr_d, double* _unknown_arr_d_prev, double* _unknown_arr_d_new, \
double* _err_d)
{
    __shared__ double rel_err;
    if(threadIdx.x == 0)
        rel_err = 0;
    int unknown_i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if(unknown_i < _n)
    {
        double x_new = (_unknown_arr_d_new[unknown_i] + _c_arr_d[unknown_i]) / _diag_arr_d[unknown_i];
        double x_old = _unknown_arr_d_old[unknown_i];
        _unknown_arr_d_old[unknown_i] = x_new;
        _unknown_arr_d_new[unknown_i] = 0;
        atomicMax(&rel_err, abs((x_new - x_old) / x_new));
    }
    __syncthreads();
    if(threadIdx.x == 0 && unknown_i < _n)
        atomicMax(_err_d, rel_err);
}