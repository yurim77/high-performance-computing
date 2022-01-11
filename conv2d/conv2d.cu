#include <stdio.h>

#define KERNEL_SIZE 5 // must be odd number 3, 5, ...
#define TILE_SIZE 8
#define BLOCK_SIZE (TILE_SIZE + (KERNEL_SIZE - 1))
#define MASK_RADIUS (KERNEL_SIZE / 2)

__constant__ float Mc[KERNEL_SIZE][KERNEL_SIZE];

void verification(const float *N, const float *M, const float *P, int Rows, int Columns) {
	int r, c, h, w;
	int row_i, col_i;
	bool equal;
	float* results;

	results = (float*)malloc(Rows * Columns * sizeof(float));
	memset(results, 0, Rows * Columns * sizeof(float));

	for (r = 0; r < Rows; r++) {
		for (c = 0; c < Columns; c++) {
			for (h = 0; h < KERNEL_SIZE; h++) {
				for (w = 0; w < KERNEL_SIZE; w++) {
					row_i = r - ((KERNEL_SIZE - 1) / 2) + h;
					col_i = c - ((KERNEL_SIZE - 1) / 2) + w;
					if ((row_i >= 0) && (row_i < Rows) && (col_i >= 0) && (col_i < Columns)) {
						results[r*Columns + c] += (M[h*KERNEL_SIZE + w] * N[row_i*Columns + col_i]);
					}
				}
			}
		}
	}

	equal = true;
	for (int i = 0; i < Rows * Columns && equal; i++) {
		if (abs(results[i] - P[i]) >= 0.001f) {
			equal = false;
			printf("NOT EQUAL!\n");
		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}

	free(results);
	return;
}

__global__ void convolution_2D_kernel(float* N, float* P, int height, int width){
	__shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	int row_i = row_o - MASK_RADIUS;
	int col_i = col_o - MASK_RADIUS;

	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)){
		Ns[ty][tx] = N[row_i * width + col_i];
	}
	else {
		Ns[ty][tx] = 0.0f;
	}

	__syncthreads();

	float output = 0.0f;
	if (ty < TILE_SIZE && tx < TILE_SIZE) {
		for (int i = 0; i < KERNEL_SIZE; i++) {
			for (int j = 0; j < KERNEL_SIZE; j++) {
				output +=  Mc[i][j] * Ns[i + ty][j + tx];
			}
		}


		if (row_o < height && col_o < width){
			P[row_o * width + col_o] = output;
		}
	}
}

void checkCUDAError(){
	cudaError_t error = cudaGetLastError();
	if (cudaSuccess != error){
		fprintf(stderr, "Cuda Error : %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char *argv[]) {
	///// Get User Input /////
    if(argc != 3){
		printf("Usage : %s <Matrix height> <Matrix width>\n", argv[0]);
		return 0;
	}

	int height = atoi(argv[1]);
    int width = atoi(argv[2]);

	size_t mat_size = height * width * sizeof(float);
	size_t kernel_size = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

	///// HOST Variables /////
	float *h_N, *h_M, *h_P_d;

	h_N = (float*)malloc(mat_size);
    h_M = (float*)malloc(kernel_size);
    h_P_d = (float*)malloc(mat_size);

	for(int i = 0; i < height * width; i++) h_N[i] = (float)(rand()%10);
    for(int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) h_M[i] = (float)(rand()%10);

	///// DEVICE Variables /////
	float *d_N, *d_P; 

	cudaMalloc((void**)&d_N, mat_size);
    cudaMalloc((void**)&d_P, mat_size);

    cudaMemcpy(d_N, h_N, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Mc, h_M, kernel_size);

	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid((width - 1) / TILE_SIZE + 1, (height - 1) / TILE_SIZE + 1);
	
	///// Execute Kernel /////
	convolution_2D_kernel<<<dim_grid, dim_block>>>(d_N, d_P, height, width);
	cudaDeviceSynchronize();

    cudaMemcpy(h_P_d, d_P, mat_size, cudaMemcpyDeviceToHost);

    checkCUDAError();

	///// Verify Result /////
	verification(h_N, h_M, h_P_d, height, width);

	///// Free Memory /////
    free(h_N);
    free(h_M);
    free(h_P_d); 

    cudaFree(d_N);
    cudaFree(d_P);

	return 0;
}