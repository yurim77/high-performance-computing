#include <immintrin.h>
#include <x86intrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>

#define ARRAY_LEN 4

// functions
void init_nxn_float_matrix();
void print_matrix(char);
void matmul_with_AVX();
void matmul_with_non_AVX();
void matmul_compare();
void usage(const char*);

// global definition of matrix
float** matrix_a;
float** matrix_b;
float** matrix_c;
__m128 matrix_d[ARRAY_LEN];

void init_nxn_float_matrix()
{
	matrix_a = (float**)calloc(ARRAY_LEN, sizeof(float*));
	matrix_b = (float**)calloc(ARRAY_LEN, sizeof(float*));
	matrix_c = (float**)calloc(ARRAY_LEN, sizeof(float*));

	for(int i=0; i<ARRAY_LEN; i++)
	{
		matrix_a[i] = (float*)aligned_alloc(ARRAY_LEN * sizeof(float), ARRAY_LEN * sizeof(float));
		matrix_b[i] = (float*)aligned_alloc(ARRAY_LEN * sizeof(float), ARRAY_LEN * sizeof(float));
		matrix_c[i] = (float*)aligned_alloc(ARRAY_LEN * sizeof(float), ARRAY_LEN * sizeof(float));
	}

	for(int i=0; i<ARRAY_LEN; i++)
	{
		for(int j=0; j<ARRAY_LEN; j++)
		{
			matrix_a[i][j] = (float)(i+j*0.1);
			matrix_b[i][j] = (float)(i+j*0.4);
		}
	}
}

void print_matrix(char matrix_name)
{	
	float** matrix;

	switch(matrix_name)
	{
		case 'a':
			matrix = matrix_a;
			break;
			
		case 'b':
			matrix = matrix_b;
			break;
			
		case 'c':
			matrix = matrix_c;
			break;
			
		case 'd':
			for(int i=0; i<ARRAY_LEN; i++)
			{
				for(int j=0; j<ARRAY_LEN; j++) printf("%-4.1lf ", matrix_d[i][j]);
				printf("\n");
			}
			return;
	}
	
	for(int i=0; i<ARRAY_LEN; i++)
	{
		for(int j=0; j<ARRAY_LEN; j++) printf("%-4.1lf ", matrix[i][j]);
		printf("\n");
	}
}

void matmul_with_AVX()
{
	// calculate matrix multiplication (AVX)
	uint64_t start_time, end_time;

	__m128 first_operand, second_operand, temp_result; 
	for(int i=0; i<ARRAY_LEN; i++) matrix_d[i] = _mm_load_ps(matrix_c[i]);     	

	start_time = __rdtsc();
	for(int i=0; i<ARRAY_LEN; i++)
	{
		for(int j=0; j<ARRAY_LEN; j++)
		{
			first_operand = _mm_broadcast_ss(&matrix_a[i][j]);
			second_operand = _mm_load_ps(matrix_b[j]);
			temp_result = _mm_mul_ps(first_operand, second_operand);
			matrix_d[i] = _mm_add_ps(temp_result, matrix_d[i]);
		}
	}
	end_time = __rdtsc();
	
	// print result
	printf("Elapsed time with AVX: %" PRIu64 "\n", end_time - start_time);
	printf("Matrix multiplication result with AVX.\n");
	print_matrix('a');
	printf("X\n");
	print_matrix('b');
	printf("=\n");
	print_matrix('d');

}

void matmul_with_non_AVX()
{
	// calculate matrix multiplication (non-AVX)
	uint64_t start_time, end_time;

	start_time = __rdtsc();
	for(int i=0; i<ARRAY_LEN; i++)
	{
		for(int j=0; j<ARRAY_LEN; j++)
		{
			for(int k=0; k<ARRAY_LEN; k++)
			{
				matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
			}
		}
	}
	end_time = __rdtsc();

	// print result
	printf("Elapsed time with non-AVX: %" PRIu64 "\n", end_time - start_time);
	printf("Matrix multiplication result with non-AVX.\n");
	print_matrix('a');
	printf("X\n");
	print_matrix('b');
	printf("=\n");
	print_matrix('c');
}

void matmul_compare()
{
	matmul_with_AVX();
	printf("\n");
	matmul_with_non_AVX();
	
	// compare AVX result with non-AVX result
	int cflag = 0;
	for(int i=0; i<ARRAY_LEN; i++)
	{
		for(int j=0; j<ARRAY_LEN; j++)
		{
			if(matrix_c[i][j] != matrix_d[i][j])
			{
				cflag = 1;
				break;
			}
		}
	}

	if(cflag) printf("\nAVX and non-AVX result are NOT the same.\n");
	else printf("\nAVX and non-AVX result are the same.\n");
}

int main(int argc, char* argv[])
{
	char opt;
	opterr = 0;
	
	init_nxn_float_matrix();
	
	while((opt = getopt(argc, argv, "v:c")) != EOF)
	{
		switch(opt)
		{
			case 'v':
				if(!strcmp(optarg,"1")) matmul_with_AVX();
				else if(!strcmp(optarg,"0")) matmul_with_non_AVX();
				else usage(argv[0]);
				break;

			case 'c':
				matmul_compare();
				break;
				
			default:
				usage(argv[0]);
				break;
		}
	}
	
	return 0;
}

void usage(const char* file_name)
{
	printf("usage : %s [OPTION] [VERSION]\n"
			" -c            compare AVX/non-AVX matrix multiplication\n"
			" -v 1          matrix multiplication with AVX\n"
			" -v 0          matrix multiplication with non-AVX\n"
			, file_name);
}

