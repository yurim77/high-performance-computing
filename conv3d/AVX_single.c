#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

void verification(float* output, float* h_P_d, int height, int width, int depth){
  int is_correct= 1;
  for(int z=0; z<depth; z++){
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        if(abs(output[z * height * width + y * width + x] - h_P_d[z * height * width + y * width + x]) >= 0.001f){
          is_correct = 0;
          break;
        }
      }
    }
  }
  if(is_correct==1) printf("Single-Threaded AVX reusult is correct!\n\n");
  else printf("Single-Threaded AVX reusult is NOT correct... try again...\n\n");
}

void conv_1d(float *in, float *aligned_out, int length, float* kernel, int KERNEL_SIZE, int PADD_RADIUS) {
   __m256 kernel_vec[KERNEL_SIZE] ;
   __m256 data_block ;

   __m256 prod ;
   __m256 acc ;

   int i, k;

   for(i = 0; i < KERNEL_SIZE; ++i) {
      kernel_vec[i] = _mm256_set1_ps(kernel[i]);
   }
   for(i = 0; i < length - 8; i+=8) {
      acc = _mm256_setzero_ps();

      for(k = -PADD_RADIUS; k <= PADD_RADIUS; ++k) {
         data_block = _mm256_loadu_ps(in + i + k);
         prod = _mm256_mul_ps(kernel_vec[k + PADD_RADIUS], data_block);
         acc = _mm256_add_ps(acc, prod);
      }

      _mm256_store_ps(aligned_out + i, acc);
   }

   // Scalar computation for the rest < 8 pixels.
   while(i != length) {
      aligned_out[i] = 0.0;
      for(k = -PADD_RADIUS; k <= PADD_RADIUS; ++k)
         aligned_out[i] += in[i + k] * kernel[k+PADD_RADIUS];
      ++i;
   }

}

void conv_2d(float *cache_in, float *cache_out, int ARRAY_HEIGHT, int ARRAY_WIDTH, float *convolution_matrix, float **lines, int KERNEL_SIZE, int PADD_RADIUS) {

  int start_row = PADD_RADIUS;
  int end_row = ARRAY_HEIGHT + PADD_RADIUS - 1;
  int start_col = PADD_RADIUS;
  int end_col = ARRAY_WIDTH + PADD_RADIUS - 1;
  int width = ARRAY_WIDTH + 2*PADD_RADIUS;

   __m256 sum1 ;
   __m256 sum2 ;
    __m256 sum3 ;
 
   for(int row = start_row; row <= end_row; ++row) {
      // Compute 1D convolutions.
      int thr_id, thr_return;
      if(KERNEL_SIZE==3){
        conv_1d(cache_in + (row - 1) * width + start_col, lines[0], width - 2, convolution_matrix, KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + row * width + start_col, lines[1], width - 2, convolution_matrix + 3,KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + (row + 1) * width + start_col, lines[2], width - 2, convolution_matrix + 6,KERNEL_SIZE,PADD_RADIUS);
   
      }else if(KERNEL_SIZE==5){
        conv_1d(cache_in + (row - 2) * width + start_col, lines[0], width - 4, convolution_matrix, KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + (row - 1) * width + start_col, lines[1], width - 4, convolution_matrix+5, KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + row * width + start_col, lines[2], width - 4, convolution_matrix + 10,KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + (row + 1) * width + start_col, lines[3], width - 4, convolution_matrix + 15,KERNEL_SIZE,PADD_RADIUS);
        conv_1d(cache_in + (row + 2) * width + start_col, lines[4], width - 4, convolution_matrix + 20,KERNEL_SIZE,PADD_RADIUS);
      }

      int i, k;
      for(i = start_col, k = 0; i <= end_col - 8; k+=8, i+=8) {

      if(KERNEL_SIZE==3){
         sum1 = _mm256_add_ps(_mm256_load_ps(&lines[0][k]), _mm256_load_ps(&lines[1][k]));
         sum2 = _mm256_add_ps(sum1, _mm256_load_ps(&lines[2][k]));
         _mm256_storeu_ps(&cache_out[row * width + i], sum2);
      }
       else if(KERNEL_SIZE==5){
         sum1 = _mm256_add_ps(_mm256_load_ps(&lines[0][k]), _mm256_load_ps(&lines[1][k]));
         sum2 = _mm256_add_ps(_mm256_load_ps(&lines[2][k]), _mm256_load_ps(&lines[3][k]));
         sum3 = _mm256_add_ps(sum1, _mm256_add_ps(_mm256_load_ps(&lines[4][k]), sum2));
         _mm256_storeu_ps(&cache_out[row * width+i], sum3);
      } 
      }

      while(i <= end_col) {
         if(KERNEL_SIZE == 5) cache_out[row * width + i] = lines[0][k] + lines[1][k] + lines[2][k] + lines[3][k] + lines[4][k];
         else cache_out[row * width + i] = lines[0][k] + lines[1][k] + lines[2][k];
        ++i;
        ++k;
      }
   }
}


void conv_3d(float* src, float* dst, float *kernel,
   int KERNEL_SIZE, int PADD_RADIUS, int ARRAY_DEPTH,  int ARRAY_HEIGHT, int ARRAY_WIDTH){
      
   int start_depth = PADD_RADIUS;
   int end_depth = ARRAY_DEPTH + PADD_RADIUS - 1;

   int PADD_ARRAY_WIDTH = ARRAY_WIDTH + PADD_RADIUS * 2;
   int PADD_ARRAY_HEIGHT = ARRAY_HEIGHT + PADD_RADIUS * 2;

   for(int depth=start_depth; depth<=end_depth; ++depth) {
      // alloc lines
      float *lines[KERNEL_SIZE];
      for(int i=0; i < KERNEL_SIZE; i++)
      {
         lines[i] =  (float*)aligned_alloc(32, PADD_ARRAY_HEIGHT * PADD_ARRAY_WIDTH * sizeof(float));
      }

      // compute 2D convolutions
      for(int j= -PADD_RADIUS; j <=PADD_RADIUS;j++){
         float* temp_dst = (float*)calloc(PADD_ARRAY_HEIGHT * PADD_ARRAY_WIDTH, sizeof(float));
         conv_2d(
            src + (depth + j) * PADD_ARRAY_HEIGHT * PADD_ARRAY_WIDTH, temp_dst,  
            ARRAY_HEIGHT, 
            ARRAY_WIDTH,
            kernel + KERNEL_SIZE * KERNEL_SIZE * (j + PADD_RADIUS),
            lines, // 2D array pointer
            KERNEL_SIZE,
            PADD_RADIUS
         );
         
         // merge result
         for(int i=PADD_RADIUS; i<PADD_ARRAY_HEIGHT - PADD_RADIUS; i++){
            for(int z=PADD_RADIUS; z<PADD_ARRAY_WIDTH - PADD_RADIUS; z++){
               dst[(depth - PADD_RADIUS) * ARRAY_HEIGHT * ARRAY_WIDTH + (i-PADD_RADIUS) * ARRAY_WIDTH + (z-PADD_RADIUS)] += temp_dst[i * PADD_ARRAY_WIDTH + z];
            }
         }
         free(temp_dst);
      }
      // free lines
      for(int i=0; i < KERNEL_SIZE; i++) free(lines[i]);
   } // end of for depth
}

int main(int argv, char** argc){
   // Read input from files //
   /* usage : ./3dconv <input file> <kernel file> <output file> */
   if(argv < 4){
      printf("Usage : ./3dconv <input file> <kernel file> <output file>\n");
      return 0;
   }else{
      // printf("-------------- Files --------------");
      // printf("\n");
      // printf("%s\n",argc[1]);
      // printf("%s\n",argc[2]);
      // printf("%s\n",argc[3]);
      // printf("\n");
   }

   FILE *input_file = fopen(argc[1], "r");
   FILE *kernel_file = fopen(argc[2], "r");
   FILE *output_file = fopen(argc[3], "r");

   int input_size=0, input_x=0, input_y=0,input_z=0;
   int output_size=0, output_x=0, output_y=0, output_z=0;
   int kernel_size=0;

   float* input;
   float* kernel;
   float* output;

   int isize=0, ksize=0, osize=0;

   /* read input */
   fscanf(input_file, "%d", &input_z);
   fscanf(input_file, "%d", &input_y);
   fscanf(input_file, "%d", &input_x);

   input_size = input_x * input_y * input_z;
   input = (float*)malloc(sizeof(float) * input_size);

   while(!feof(input_file)){  
      fscanf(input_file,"%f", &input[isize++]);
   }

   /* read kernel */
   fscanf(kernel_file, "%d", &kernel_size);

   kernel = (float*)malloc(sizeof(float) * kernel_size * kernel_size * kernel_size);

   while(!feof(kernel_file)){  
      fscanf(kernel_file,"%f", &kernel[ksize++]);
   }

   /* read output */
   fscanf(output_file, "%d", &output_z);
   fscanf(output_file, "%d", &output_y);
   fscanf(output_file, "%d", &output_x);

   output_size = output_x*output_y*output_z;
   output = (float*)malloc(sizeof(float)*output_size);

   while(!feof(output_file)){  
      fscanf(output_file,"%f", &output[osize++]);
   }
  
   fclose(input_file);
   fclose(kernel_file);
   fclose(output_file);

   int KERNEL_SIZE = kernel_size;
   int ARRAY_HEIGHT = input_y;
   int ARRAY_WIDTH = input_x;
   int ARRAY_DEPTH = input_z;

   int PADD_RADIUS = (KERNEL_SIZE / 2);
   int PADD_ARRAY_HEIGHT =(ARRAY_HEIGHT + (PADD_RADIUS * 2));
   int PADD_ARRAY_WIDTH =(ARRAY_WIDTH + (PADD_RADIUS * 2));
   int PADD_ARRAY_DEPTH =(ARRAY_DEPTH + (PADD_RADIUS * 2));

   // convolution src mat, dst mat
   float* src = (float*)calloc(PADD_ARRAY_DEPTH * PADD_ARRAY_HEIGHT * PADD_ARRAY_WIDTH, sizeof(float));
   float* dst = (float*)calloc(ARRAY_DEPTH * ARRAY_HEIGHT * ARRAY_WIDTH, sizeof(float));
  
   // padding src mat
   int idx, idxx;
   for(int k=0; k<PADD_ARRAY_DEPTH; k++){   
      for(int i=0; i < PADD_ARRAY_HEIGHT; i++){
         for(int j=0; j < PADD_ARRAY_WIDTH; j++){
            idx = k * PADD_ARRAY_WIDTH * PADD_ARRAY_HEIGHT + i * PADD_ARRAY_WIDTH + j;
            idxx = (k - PADD_RADIUS) * ARRAY_WIDTH * ARRAY_HEIGHT + (i - PADD_RADIUS) * ARRAY_WIDTH + (j - PADD_RADIUS);
            if((k < PADD_RADIUS) || (k >= PADD_ARRAY_DEPTH - PADD_RADIUS)) src[idx] = 0.0f;
            else if ((i < PADD_RADIUS) || (i >= PADD_ARRAY_HEIGHT - PADD_RADIUS)) src[idx] = 0.0f;
            else if ((j < PADD_RADIUS) || (j >= PADD_ARRAY_WIDTH - PADD_RADIUS)) src[idx] = 0.0f;
            else src[idx] = input[idxx];
         }
      }
   }

   // convolution
   clock_t start, finish;
   double duration;
   start = clock();

   conv_3d(src, dst, kernel,
         KERNEL_SIZE, PADD_RADIUS, ARRAY_DEPTH, ARRAY_HEIGHT, ARRAY_WIDTH);
   long long end = __rdtsc();
   finish = clock();
   duration = (double)(finish-start)/CLOCKS_PER_SEC;
   printf("Single-Threaded AVX Execution Time: %.3lf ms\n", duration * 1000);


   // vertify result
   verification(output, dst, ARRAY_HEIGHT, ARRAY_WIDTH, ARRAY_DEPTH);
  
   // free memory
   free(src);
   free(dst);

   free(input);
   free(output);
   free(kernel);

   return 0;
}