## **3D Convolution using CUDA & AVX**
- co-developer : [dmstmdrbs](https://github.com/dmstmdrbs)

### Description
- 3D Convolution을 CUDA 프로그래밍과 AVX(Advanced Vector Extension)를 이용하여 구현한 프로젝트입니다.
- CUDA, single thread AVX, multi thread AVX의 3D 컨볼루션 계산 속도를 비교할 수 있습니다.

### Files
- CUDA.cu : CUDA 프로그래밍을 이용한 3D Convolution 구현
- AVX_single.c : 싱글 쓰레드를 기반으로 AVX를 이용한 3D Convolution 구현
- AVX_multi.c : 멀티 쓰레드를 기반으로 AVX를 이용한 3D Convolution 구현
- AVX_multi_threadpool.c : 쓰레드 풀을 이용한 멀티 쓰레드를 기반으로 AVX를 이용한 3D Convolution 구현 (AVX_multi.c의 성능을 개선한 버전)
- Makefile : compile & make object file

### Usage
- 바이너리 파일 만들기 / 삭제하기
```shell
$ make test1
$ make test2
$ make test3
$ make test4
$ make test5

$ make clean
```

- 바이너리 파일 실행하기

```shell
# Execute binary file
# Usage : Usage : ./3dconv <input file> <kernel file> <output file>
$ ./3dconv ./sample/test1/input.txt ./sample/test1/kernel.txt ./sample/test1/output.txt

# Execution Result
-------------- Files --------------
./sample/test1/input.txt
./sample/test1/kernel.txt
./sample/test1/output.txt

input size (z * y * x) : 64 * 64 * 64
kernel size (z * y * x) : 3 * 3 * 3
output size (z * y * x) : 64 * 64 * 64

CUDA Execution Time : 1.292 ms
CUDA reusult is correct!

Single-Threaded AVX Execution Time: 34.602 ms
Single-Threaded AVX reusult is correct!

Multi-Threaded AVX Execution Time: 64.334 ms
Multi-Threaded AVX reusult is correct!

Multi-Threaded using Thread Pool AVX Execution Time: 61.882 ms
Multi-Threaded AVX reusult is correct!

```
