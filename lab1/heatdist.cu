/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s) that you need to write too. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 

#define TILE_WIDTH 10
// #define DEBUG_PRINT

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);
__global__ void heat_dist_kernel(float* _mat1, float* _mat2, int _width);
__global__ void my_device_cpy_kernel(float* _mat1, float* _mat2, int _width);


/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
 
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  for(i = 0; i < N-1; i++)
    playground[index(N-1,i,N)] = 150;
  
  if( !type_of_device ) // The CPU sequential version
  {  
    start = clock();
    seq_heat_dist(playground, N, iterations);
    end = clock();
  }
  else  // The GPU version
  {
     start = clock();
     gpu_heat_dist(playground, N, iterations); 
     end = clock();    
  }
  
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
__global__ void heat_dist_kernel(float* _mat1, float* _mat2, int _width)
{
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  if(row > 0 && row < _width - 1 && col > 0 && col < _width - 1)
  {
    _mat2[index(row, col, _width)] = 
      (_mat1[index(row - 1, col, _width)]
      + _mat1[index(row, col - 1, _width)]
      + _mat1[index(row, col + 1, _width)]
      + _mat1[index(row + 1, col, _width)])/4.0;   
  }
}

__global__ void my_device_cpy_kernel(float* _mat1, float* _mat2, int _width)
{
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int i = index(row, col, _width);
  _mat1[i] = _mat2[i];
}

void  gpu_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  float *mat1_d, *mat2_d;
  unsigned int num_bytes = N*N*sizeof(float);
  cudaMalloc((void**)&mat1_d, num_bytes);
  cudaMalloc((void**)&mat2_d, num_bytes);
  cudaMemcpy(mat1_d, playground, num_bytes, cudaMemcpyHostToDevice);
  dim3 dim_grid(N/TILE_WIDTH, N/TILE_WIDTH);
  dim3 dim_block(TILE_WIDTH, TILE_WIDTH);
  for(int i = 0; i < iterations; ++ i)
  {
    heat_dist_kernel<<<dim_grid, dim_block>>>(mat1_d, mat2_d, N);
    my_device_cpy_kernel<<<dim_grid, dim_block>>>(mat1_d, mat2_d, N);
  }
  cudaMemcpy(playground, mat1_d, num_bytes, cudaMemcpyDeviceToHost);
#ifdef DEBUG_PRINT
  for(int i = 0; i < N; ++ i)
  {
    for(int j = 0; j < N; ++ j)
      printf("%f ", playground[i*N + j]);
    printf("\n");
  }
#endif
}


