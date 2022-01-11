#include<stdio.h>

#define TILE_WIDTH 16

// matrix multiplication (tiled, kernel) 
__global__ void matrixMulTiledKernel(float *d_M, float *d_N, float *d_P, int height, int width, int temp){
    __shared__ float d_Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float d_Ns[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    if((row < height) && (col < width)){
        for (int l = 0; l<temp/TILE_WIDTH; ++l){ 
            d_Ms[ty][tx] = d_M[row * temp + l * TILE_WIDTH + tx];
            d_Ns[ty][tx] = d_N[(l * TILE_WIDTH + ty) * width + col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k){
                Pvalue += d_Ms[ty][k] * d_Ns[k][tx];
            }
            __syncthreads();
        }
        d_P[row * width + col] = Pvalue;
    }
}

// matrix multiplication (non-tiled, kernel)
__global__ void matrixMulKernel(float *d_M, float *d_N, float *d_P, int height, int width, int temp){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;

    if((row < height) && (col < width)){
        Pvalue = 0;
        for (int k = 0; k < temp; ++k){
            Pvalue += d_M[row * temp + k] * d_N[k * width + col];   
        }
        d_P[row * width + col] = Pvalue;
    }
}

// matrix multiplication (host)
void matrixMulHost(float *h_M, float *h_N, float *h_P, int height, int width, int temp){
    float Pvalue = 0;

    for(int row = 0; row < height; ++row){
        for(int col = 0; col < width; ++col){
            Pvalue = 0;
            for(int k = 0; k < temp; ++k){
                Pvalue += h_M[row * temp + k] * h_N[k * width + col];
            }
            h_P[row * width + col] = Pvalue;
        }
    }
}

// print matrix
void printMatrix(float *matrix, int height, int width){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            printf("%.1lf ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

// compare matrix A and B
void compareMatrices(float *A, float *B, int height, int width){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            if(A[i * width + j] != B[i * width + j]){
                printf("Two matrices do NOT match each other...\n");
                return;
            }
        }
    }
    printf("Two matrices match each other!\n");
}

// check CUDA error
void checkCUDAError(){
	cudaError_t error = cudaGetLastError();
	if (cudaSuccess != error){
		fprintf(stderr, "Cuda Error : %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char* argv[]){
    ///// Variables for Mearusing Execution Time /////
    cudaEvent_t start_time, end_time;
    float time_ms = 0;

    ///// Get User Input /////
    if(argc != 5){
		printf("Usage : %s <Matrix M height> <Matrix M width> <Matrix N height> <Matrix N width>\n", argv[0]);
		return 0;
	}

	int M_height = atoi(argv[1]);
    int M_width = atoi(argv[2]);
    int N_height = atoi(argv[3]);
    int N_width = atoi(argv[4]);

    if(M_width != N_height){
        printf("Matrix M height and Matrix N width mush be same.\n");
        return 0;
    }

    if(M_height < TILE_WIDTH || M_width < TILE_WIDTH || N_height < TILE_WIDTH || N_width < TILE_WIDTH){
        printf("Matrix height or width must be greater than TILE_WIDTH : %d \n", TILE_WIDTH);
        return 0;
    }

    ///// Initialize - HOST (CPU) /////
    float *h_M, *h_N, *h_P, *h_P_d, *h_P_dt;
    float *d_M, *d_N, *d_P, *dt_P;

    size_t M_size = M_height * M_width * sizeof(float);
    size_t N_size = N_height * N_width * sizeof(float);
    size_t P_size = M_height * N_width * sizeof(float);

    h_M = (float*)malloc(M_size);
    h_N = (float*)malloc(N_size);
    h_P = (float*)malloc(P_size);
    h_P_d = (float*)malloc(P_size);
    h_P_dt = (float*)malloc(P_size);

    for(int i = 0; i < M_height * M_width; i++) h_M[i] = (float)(rand()%10);
    for(int i = 0; i < N_height * N_width; i++) h_N[i] = (float)(rand()%10);
    
    ///// Execute - HOST (CPU) /////
    // matrixMulHost(h_M, h_N, h_P, M_height, N_width, M_width);

    ///// Initialize - DEVICE (GPU) /////
    cudaMalloc((void**)&d_M, M_size);
    cudaMalloc((void**)&d_N, N_size);
    cudaMalloc((void**)&d_P, P_size);
    cudaMalloc((void**)&dt_P, P_size);

    cudaMemcpy(d_M, h_M, M_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, N_size, cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid(ceil(N_width/TILE_WIDTH*1.0), ceil(M_height/TILE_WIDTH*1.0), 1);

    cudaEventCreate(&start_time);
	cudaEventCreate(&end_time);

    ///// Execute - Tilted Kernel, DEVICE (GPU) /////
    cudaEventRecord(start_time, 0);
    matrixMulTiledKernel<<<dim_grid, dim_block>>>(d_M, d_N, dt_P, M_height, N_width, M_width);
    cudaEventRecord(end_time, 0);
	cudaEventSynchronize(end_time);

    cudaDeviceSynchronize();
    cudaMemcpy(h_P_dt, dt_P, P_size, cudaMemcpyDeviceToHost);
    checkCUDAError();

    cudaEventElapsedTime(&time_ms, start_time, end_time);
    printf("Execution time for Tilted Kernel : %.5f ms\n", time_ms);

    ///// Execute - Non-Tilted Kernel, DEVICE (GPU) /////
    cudaEventRecord(start_time, 0);
    matrixMulKernel<<<dim_grid, dim_block>>>(d_M, d_N, d_P, M_height, N_width, M_width);
    cudaEventRecord(end_time, 0);
	cudaEventSynchronize(end_time);

    cudaDeviceSynchronize();
    cudaMemcpy(h_P_d, d_P, P_size, cudaMemcpyDeviceToHost);
    checkCUDAError();

    cudaEventElapsedTime(&time_ms, start_time, end_time);
    printf("Execution time for Non-Tilted Kernel : %.5f ms\n", time_ms);

    ///// Compare Results /////
    compareMatrices(h_P_d, h_P_dt, M_height, N_width);

    ///// Free Memory /////
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_d);
    free(h_P_dt);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaFree(dt_P);

    return 0;
}
