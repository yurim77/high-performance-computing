## **2D convolution**

### Description
* 병렬 처리를 위해 제작된 NVIDIA 아키텍처 인 CUDA를 이용하여 2차원 컨볼루션을 구현한 프로그램입니다.
* 매트릭스의 높이와 너비는 사용자 입력을 받습니다. 
* 커널 크기는 #define을 통해 상수(KERNEL_SIZE)로 선언되어 있습니다. KERNEL_SIZE는 수정 가능하나 홀수이어야 합니다.
* 디바이스에서 계산한 결과는 유저에서 검증하고 값이 옳은지 아닌지 메시지로 출력합니다.

### Environment
* Device : GeForce GTX 1080 
* Cuda Driver Version / Runtime Version : 11.0 / 10.2

### Files
* conv2d.cu : 2차원 컨볼루션을 구현한 CUDA C 코드입니다.
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
# Usage: ./conv2d <matrix height> <matrix width>

# Output example
$ ./conv2d 128 128
Results are equal!