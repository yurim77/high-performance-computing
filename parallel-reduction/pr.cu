#include <stdio.h>

__global__
void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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
	cudaMalloc((void**)&d_out, sizeof(int) * (vector_size / n_threads + 1));

	for (int i = 0; i < vector_size; i++) h_in[i] = 1;

	cudaError_t err = cudaMemcpy(d_in, h_in, sizeof(int) * vector_size, cudaMemcpyHostToDevice);

	cudaEventCreate(&t1);
	cudaEventCreate(&t2);

	cudaEventRecord(t1, 0);

	reduce0<<<vector_size / n_threads, n_threads, n_threads * sizeof(int)>>>(d_in, d_out);
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
