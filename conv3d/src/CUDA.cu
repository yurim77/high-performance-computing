#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define _CRT_SECURE_NO_WARNINGS 

#define MAX_KERNEL_SIZE 125

__constant__ float Mc[MAX_KERNEL_SIZE];

void verification(float* output, float* h_P_d, int height, int width, int depth){
  int is_correct= 1;
  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        if(abs(output[z * height * width + y * width + x] - h_P_d[z * height * width + y * width + x]) >= 0.0001f){
          is_correct = 0;
          break;
        }
      }
    }
  }
  if(is_correct==1) printf("CUDA reusult is correct!\n\n");
  else printf("CUDA result is NOT correct... try again...\n\n");
}

void checkCUDAError(){
  cudaError_t error = cudaGetLastError();
  if (cudaSuccess != error){
    fprintf(stderr, "Cuda Error : %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

__global__ void convolution_2D_kernel(float* N, float* P, int height, int width, int depth, 
                                      const int KERNEL_SIZE, const int TILE_SIZE, const int BLOCK_SIZE, const int MASK_RADIUS){
  extern __shared__ float Ns[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x_o = blockIdx.x * TILE_SIZE + tx;
  int y_o = blockIdx.y * TILE_SIZE + ty;
  int z_o = blockIdx.z * TILE_SIZE + tz;

  int x_i = x_o - MASK_RADIUS;
  int y_i = y_o - MASK_RADIUS;
  int z_i = z_o - MASK_RADIUS;

  if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height) && (z_i >= 0) && (z_i < depth)){
    Ns[tz * BLOCK_SIZE * BLOCK_SIZE + ty * BLOCK_SIZE + tx] = N[z_i * width * height + y_i * width + x_i];
  }
  else {
    Ns[tz * BLOCK_SIZE * BLOCK_SIZE + ty * BLOCK_SIZE + tx] = 0.0f;
  }
  __syncthreads();

  float output = 0.0f;
  if (tz < TILE_SIZE && ty < TILE_SIZE && tx < TILE_SIZE) {
    for (int z = 0; z < KERNEL_SIZE; z++) {
      for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
          output +=  Mc[z * KERNEL_SIZE * KERNEL_SIZE + y * KERNEL_SIZE + x] * 
                        Ns[(z + tz) * BLOCK_SIZE * BLOCK_SIZE + (y + ty) * BLOCK_SIZE + (x + tx)];
        }
      } 
    } // end of for z

    if (z_o < depth && y_o < height && x_o < width){
      P[z_o * height * width + y_o * width + x_o] = output;
    }
  } // end of if
}

int main(int argv, char** argc){
  // Read input from files //
  /* usage : ./3dconv <input file> <kernel file> <output file> */
  if(argv < 4){
    printf("Usage : ./3dconv <input file> <kernel file> <output file>\n");
    return 0;
  }else{
    printf("-------------- Files --------------");
    printf("\n");
    printf("%s\n",argc[1]);
    printf("%s\n",argc[2]);
    printf("%s\n",argc[3]);
    printf("\n");
  }

  FILE *input_file = fopen(argc[1], "r");
  FILE *kernel_file = fopen(argc[2], "r");
  FILE *output_file = fopen(argc[3], "r");

  int input_size=0, input_x=0, input_y=0, input_z=0;
  int output_size=0, output_x=0, output_y=0, output_z=0;
  int _kernel_size=0, kernel_len=0;

  float* input;
  float* output;
  float* kernel;

  int isize=0, ksize=0, osize=0;

  /* read input */
  fscanf(input_file, "%d", &input_z);
  fscanf(input_file, "%d", &input_y);
  fscanf(input_file, "%d", &input_x);
  printf("input size (z * y * x) : %d * %d * %d\n", input_z, input_y, input_x);

  input_size = input_x*input_y*input_z;
  input = (float*)malloc(sizeof(float)*input_size);

  while(!feof(input_file)){  
    fscanf(input_file,"%f", &input[isize++]);
  }

  /* read kernel */
  fscanf(kernel_file, "%d", &kernel_len);
  printf("kernel size (z * y * x) : %d * %d * %d\n", kernel_len, kernel_len, kernel_len);

  _kernel_size = kernel_len*kernel_len*kernel_len;
  kernel = (float*)malloc(sizeof(float)*_kernel_size);

  while(!feof(kernel_file)){  
    fscanf(kernel_file,"%f", &kernel[ksize++]);
  }

  /* read output */
  fscanf(output_file, "%d", &output_z);
  fscanf(output_file, "%d", &output_y);
  fscanf(output_file, "%d", &output_x);
  printf("output size (z * y * x) : %d * %d * %d\n", output_z, output_y, output_x);

  output_size = output_x*output_y*output_z;
  output = (float*)malloc(sizeof(float)*output_size);

  while(!feof(output_file)){  
    fscanf(output_file,"%f", &output[osize++]);
  }
  
  fclose(input_file);
  fclose(kernel_file);
  fclose(output_file);
  
  // 3D Convolution Host code //
  const int KERNEL_SIZE  = kernel_len;
  const int TILE_SIZE = 4;
  const int BLOCK_SIZE = (TILE_SIZE + (KERNEL_SIZE - 1));
  const int MASK_RADIUS = (KERNEL_SIZE / 2);

  int height = input_y;
  int width = input_x;
  int depth = input_z;

  size_t mat_size = depth * height * width * sizeof(float);
  size_t kernel_size = kernel_len *  kernel_len * kernel_len * sizeof(float);

  float *h_P_d = (float*)malloc(mat_size); // host 
  float *d_N, *d_P; // device

  cudaEvent_t hd_memcpyStart, hd_memcpyEnd, kernelStart, kernelEnd, dh_memcpyStart, dh_memcpyEnd;
  float hd_memcpyTime=0, kernelTime=0, dh_memcpyTime=0;

  cudaEventCreate(&hd_memcpyStart);
  cudaEventCreate(&hd_memcpyEnd);
  cudaEventCreate(&kernelStart);
  cudaEventCreate(&kernelEnd);
  cudaEventCreate(&dh_memcpyStart);
  cudaEventCreate(&dh_memcpyEnd);

  cudaMalloc((void**)&d_N, mat_size);
  cudaMalloc((void**)&d_P, mat_size);


  cudaEventRecord(hd_memcpyStart, 0);
  cudaMemcpy(d_N, input, mat_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, kernel, kernel_size);
  cudaEventRecord(hd_memcpyEnd, 0);
  cudaEventSynchronize(hd_memcpyEnd);

  cudaEventElapsedTime(&hd_memcpyTime, hd_memcpyStart, hd_memcpyEnd);
  

  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 dim_grid((width - 1) / TILE_SIZE + 1, (height - 1) / TILE_SIZE + 1, (depth - 1) / TILE_SIZE + 1);
  
  // Execute Kernel //
  cudaEventRecord(kernelStart, 0);
  convolution_2D_kernel<<<dim_grid, dim_block, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(float)
                        >>>(d_N, d_P, height, width, depth, KERNEL_SIZE, TILE_SIZE, BLOCK_SIZE, MASK_RADIUS);
  cudaEventRecord(kernelEnd, 0);
  cudaEventSynchronize(kernelEnd);
  cudaEventElapsedTime(&kernelTime, kernelStart, kernelEnd);
  

  cudaEventRecord(dh_memcpyStart, 0);
  cudaDeviceSynchronize();
  cudaMemcpy(h_P_d, d_P, mat_size, cudaMemcpyDeviceToHost);

  cudaEventRecord(dh_memcpyEnd, 0);
  cudaEventSynchronize(dh_memcpyEnd);

  cudaEventElapsedTime(&dh_memcpyTime, dh_memcpyStart, dh_memcpyEnd);
  
  checkCUDAError();

  // printf("Memcpy Host to Device ms time : %f\n", hd_memcpyTime);
  // printf("Kernel Execution ms time : %f\n", kernelTime);
  // printf("Memcpy Device to Host ms time : %f\n", dh_memcpyTime);
  printf("\nCUDA Execution Time : %.3f ms\n", (hd_memcpyTime + kernelTime + dh_memcpyTime));

  // Verify Result //
  verification(output, h_P_d, height,width, depth);

  // Free Memory //
  free(input);
  free(kernel);
  free(h_P_d); 
  free(output);

  cudaFree(d_N);
  cudaFree(d_P);
}