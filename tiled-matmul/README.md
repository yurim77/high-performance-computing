## **Locality and Tiled Matrix Multiplication**

### Description
* nVidia GPU에서 병렬 처리를 위해 제작된 NVIDIA 아키텍처 인 CUDA를 이용하여 행렬 곱을 구현한 프로그램입니다.
* 곱해지는 두 행렬, A와 B의 가로 길이와 세로 길이를 지정할 수 있습니다.
* 행렬 곱 계산을 위해 매번 글로벌 메모리에 접근하는 non-tiled kernel version 과 쉐어드 메모리에 값을 캐싱하는 tiled kernel version 의 연산 속도를 비교할 수 있습니다.
S
### Environment
* Device : GeForce GTX 1080 
* Cuda Driver Version / Runtime Version : 11.0 / 10.2

### Files
* matmul.cu : 행렬 곱을 계산하는 CUDA C 코드입니다.
* Makefile : 빌드 명령어를 포함하는 파일입니다.

### Usage
* 바이너리 파일 만들기 / 삭제하기
```shell
# How to Build?
$ make all
$ make clean
```

* 바이너리 파일 실행하기
```shell
# Execute binary file

# Usage: ./matmul <Matrix M height> <Matrix M width> <Matrix N height> <Matrix N width>

# Matrix M height and Matrix N width must be same.
# Matrix height and width must be greater than TILE_WIDTH(default = 16).

# Output example
$ ./matmul 256 256 256 256
Execution Time for Tiled Kernel : 0.04899 ms
Execution Time for Non-Tiled Kernel : 0.08592 ms
Two matrices match each other!