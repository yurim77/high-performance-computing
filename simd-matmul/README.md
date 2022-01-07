## **SIMD Matrix Multiplication**

### Description
* 4x4 행렬의 곱셈을 벡터 방식과 스칼라 방식을 사용해 구현한 프로그램입니다.
* 벡터 방식과 스칼라 방식의 연산 결과와 속도를 비교할 수 있습니다.
* 벡터 방식은 Advanced Vector eXtensions (AVX)를 사용합니다.

### Files
* lab.c : 벡터 방식과 스칼라 방식의 행렬 연산 코드가 구현되어 있는 파일입니다.
* Makefile : 빌드 명령어를 포함하는 파일입니다.


### Usage
* 바이너리 파일 만들기 / 삭제하기
```shell
# How to Build?
$ make all
gcc -mavx2 -o matmul lab1.c
successfully completed.

$ make clean
rm -f matmul
successfully completed.
```

* 바이너리 파일 실행하기
```shell
# Execute binary file with input arguments.
# ./matmul [OPTION] [VERSION]
# ./matmul -c   : compare AVX/non-AVX matrix multiplication
# ./matmul -v 1 : matrix multiplication with AVX
# ./matmul -v 0 : matrix multiplication with non-AVX

# Output example (1)
$ ./matmul -v 1
Elapsed time with AVX: 1085
Matrix multiplication result with AVX.
0.0  0.1  0.2  0.3
1.0  1.1  1.2  1.3
2.0  2.1  2.2  2.3
3.0  3.1  3.2  3.3
X
0.0  0.4  0.8  1.2
1.0  1.4  1.8  2.2
2.0  2.4  2.8  3.2
3.0  3.4  3.8  4.2
=
1.4  1.6  1.9  2.1
7.4  9.2  11.1 12.9
13.4 16.8 20.3 23.7
19.4 24.4 29.5 34.5

# Output example (2)
$ ./matmul -c
Elapsed time with AVX: 1127
Matrix multiplication result with AVX.
0.0  0.1  0.2  0.3
1.0  1.1  1.2  1.3
2.0  2.1  2.2  2.3
3.0  3.1  3.2  3.3
X
0.0  0.4  0.8  1.2
1.0  1.4  1.8  2.2
2.0  2.4  2.8  3.2
3.0  3.4  3.8  4.2
=
1.4  1.6  1.9  2.1
7.4  9.2  11.1 12.9
13.4 16.8 20.3 23.7
19.4 24.4 29.5 34.5

Elapsed time with non-AVX: 2437
Matrix multiplication result with non-AVX.
0.0  0.1  0.2  0.3
1.0  1.1  1.2  1.3
2.0  2.1  2.2  2.3
3.0  3.1  3.2  3.3
X
0.0  0.4  0.8  1.2
1.0  1.4  1.8  2.2
2.0  2.4  2.8  3.2
3.0  3.4  3.8  4.2
=
1.4  1.6  1.9  2.1
7.4  9.2  11.1 12.9
13.4 16.8 20.3 23.7
19.4 24.4 29.5 34.5

AVX and non-AVX result are the same.
```
