#include <stdio.h>
#include <stdlib.h>

__device__ void warpReduce(volatile int* sdata, int tid){
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__ void reduce0(int *g_idata, int *g_odata){
  extern __shared__ int sdata[]; 

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  for(unsigned int s = blockDim.x/2; s > 32; s >>= 1)
  {
    if(tid < s) sdata[tid] += sdata[tid + s];   
    __syncthreads();
  }
  
  if(tid < 32) warpReduce(sdata,tid);
  if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
  int *h_in, h_out, *d_in, *d_out;
  size_t vector_size;
  float time_ms = 0;
  int n_threads = 1024;

  cudaEvent_t t1, t2;

  vector_size = 1 << 20;

  h_in = (int*)malloc(sizeof(int) * vector_size);
  cudaMalloc((void**)&d_in, sizeof(int) * vector_size);
  cudaMalloc((void**)&d_out, sizeof(int) * (vector_size/n_threads + 1));

  for (int i = 0; i < vector_size; i++) h_in[i] = 1;

  cudaError_t err = cudaMemcpy(d_in, h_in, sizeof(int) * vector_size, cudaMemcpyHostToDevice);

  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1, 0);

  reduce0<<<vector_size / (2 * n_threads), n_threads, n_threads * sizeof(int)>>>(d_in, d_out);
  reduce0<<<1, n_threads, n_threads * sizeof(int)>>>(d_out, d_out + (vector_size / n_threads));


  cudaEventRecord(t2, 0);
  cudaEventSynchronize(t2);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // somthing's gone wrong
    // print out the CUDA error as a string
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));

    // we can't recover from the error -- exit the program
    return 1;
  }

  cudaEventElapsedTime(&time_ms, t1, t2);

  cudaMemcpy(&h_out, d_out + (vector_size / n_threads), sizeof(int), cudaMemcpyDeviceToHost);
  printf("Execution time for reduce0: %.2f ms\n", time_ms);
  printf("results: %d\n", h_out);

  cudaFree(d_out);
  cudaFree(d_in);
  free(h_in);
  
  return 0;
}
