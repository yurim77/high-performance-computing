export void min_func(
	uniform int N,
	uniform int x[],
	uniform int min_value[])
{
	int local_min_value = N;

	// assume N % programCount = 0
	for (uniform int i=0; i<N; i+=programCount)
	{
		int idx = i + programIndex;		
		if(local_min_value > x[idx]) local_min_value = x[idx];
	}
	*min_value = reduce_min(local_min_value);
}

export void max_func(
        uniform int N,
        uniform int x[],
        uniform int max_value[])
{
	int local_max_value = -1;

	// assume N % programCount = 0
	for (uniform int i=0; i<N; i+=programCount)
	{
		int idx = i + programIndex;
		if(local_max_value < x[idx]) local_max_value = x[idx];
	}
	*max_value = reduce_max(local_max_value);
}

