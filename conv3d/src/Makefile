test1: AVX_multi.c AVX_single.c CUDA.cu AVX_multi_threadpool.c ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt
	@gcc -mavx2 -o AVX_single AVX_single.c
	@gcc -mavx2 -o AVX_multi AVX_multi.c -pthread
	@gcc -mavx2 -o AVX_multi_threadpool AVX_multi_threadpool.c -pthread
	@nvcc CUDA.cu -o CUDA

	@./CUDA ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt
	@./AVX_single ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt
	@./AVX_multi ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt
	@./AVX_multi_threadpool ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt

test2: AVX_multi.c AVX_single.c CUDA.cu AVX_multi_threadpool.c ./sample/test2/input.txt ./sample/test2/kernel.txt ./sample/test2/output.txt
	@gcc -mavx2 -o AVX_single AVX_single.c
	@gcc -mavx2 -o AVX_multi AVX_multi.c -pthread
	@gcc -mavx2 -o AVX_multi_threadpool AVX_multi_threadpool.c -pthread
	@nvcc CUDA.cu -o CUDA

	@./CUDA ./sample/test2/input.txt ./sample/test2/kernel.txt ./sample/test2/output.txt
	@./AVX_single ./sample/test2/input.txt ./sample/test2/kernel.txt ./sample/test2/output.txt
	@./AVX_multi ./sample/test2/input.txt ./sample/test2/kernel.txt ./sample/test2/output.txt
	@./AVX_multi_threadpool ./sample/test2/input.txt ./sample/test2/kernel.txt ./sample/test2/output.txt

test3: AVX_multi.c AVX_single.c CUDA.cu AVX_multi_threadpool.c ./sample/test3/input.txt ./sample/test3/kernel.txt ./sample/test3/output.txt
	@gcc -mavx2 -o AVX_single AVX_single.c
	@gcc -mavx2 -o AVX_multi AVX_multi.c -pthread
	@gcc -mavx2 -o AVX_multi_threadpool AVX_multi_threadpool.c -pthread
	@nvcc CUDA.cu -o CUDA

	@./CUDA ./sample/test3/input.txt ./sample/test3/kernel.txt ./sample/test3/output.txt
	@./AVX_single ./sample/test3/input.txt ./sample/test3/kernel.txt ./sample/test3/output.txt
	@./AVX_multi ./sample/test3/input.txt ./sample/test3/kernel.txt ./sample/test3/output.txt
	@./AVX_multi_threadpool ./sample/test3/input.txt ./sample/test3/kernel.txt ./sample/test3/output.txt

test4: AVX_multi.c AVX_single.c CUDA.cu AVX_multi_threadpool.c ./sample/test4/input.txt ./sample/test4/kernel.txt ./sample/test4/output.txt
	@gcc -mavx2 -o AVX_single AVX_single.c
	@gcc -mavx2 -o AVX_multi AVX_multi.c -pthread
	@gcc -mavx2 -o AVX_multi_threadpool AVX_multi_threadpool.c -pthread
	@nvcc CUDA.cu -o CUDA

	@./CUDA ./sample/test4/input.txt ./sample/test4/kernel.txt ./sample/test4/output.txt
	@./AVX_single ./sample/test4/input.txt ./sample/test4/kernel.txt ./sample/test4/output.txt
	@./AVX_multi ./sample/test4/input.txt ./sample/test4/kernel.txt ./sample/test4/output.txt
	@./AVX_multi_threadpool ./sample/test4/input.txt ./sample/test4/kernel.txt ./sample/test4/output.txt

test5: AVX_multi.c AVX_single.c CUDA.cu AVX_multi_threadpool.c ./sample/test5/input.txt ./sample/test5/kernel.txt ./sample/test5/output.txt
	@gcc -mavx2 -o AVX_single AVX_single.c
	@gcc -mavx2 -o AVX_multi AVX_multi.c -pthread
	@gcc -mavx2 -o AVX_multi_threadpool AVX_multi_threadpool.c -pthread
	@nvcc CUDA.cu -o CUDA

	@./CUDA ./sample/test5/input.txt ./sample/test5/kernel.txt ./sample/test5/output.txt
	@./AVX_single ./sample/test5/input.txt ./sample/test5/kernel.txt ./sample/test5/output.txt
	@./AVX_multi ./sample/test5/input.txt ./sample/test5/kernel.txt ./sample/test5/output.txt
	@./AVX_multi_threadpool ./sample/test5/input.txt ./sample/test5/kernel.txt ./sample/test5/output.txt

clean: 
	@rm AVX_single AVX_multi AVX_multi_threadpool CUDA