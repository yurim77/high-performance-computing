#include "minmax_ispc.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

int main(int argc, char** argv) 
{
	int N = 1024;
	int* x = new int[N];
	int min_value = 9999;
	int max_value = -1;
	
	// initialize x : 0 to 1023
	for(int i=0; i<1024; i++) x[i] = i;
	
	// execute ispc
	ispc::min_func(N, x, &min_value);
	ispc::max_func(N, x, &max_value);

	// print result
	printf("min value is %d\n", min_value);
	printf("max value is %d\n", max_value);
	
	return 0;
}
