## **POSIX Thread Programming**

### Description
* pthread API를 이용하여 Game-of-Life 문제를 병렬처리 하는 프로그램입니다.
* 시퀀셜, 싱글 스레드, 멀티 스레드 방식으로 실행 가능합니다.
* sample_inputs 폴더에는 총 5개의 입력 예제가 포함되어있습니다.

### Files
* glife.cpp : 메인 함수가 정의되어있는 c++ 코드입니다.
* glife.h : glife.cpp 파일에 필요한 헤더입니다.
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

# Usage: ./glife <input file> <display> <nprocs> <N of generation> <width> <height>
#       display = 1: Dump results
#       nprocs = 0:  Running sequentially
#       nprocs = 1:  Running on a single thread
#       nprocs > 1:  Running on multiple threads
# 단, input file에 적힌 모든 (행, 열) 값보다 <height>, <width> 가 커야합니다.
# 단, <nprocs> 값보다 <height> 값이 커야 합니다. (스레드들이 행 단위로 병렬처리 합니다.)

# Output exaple (1)
$ ./glife ./sample_inputs/simple1_4_4 0 2 1 4 4
Execution time(seconds): 0.000383
Program end...

# Output exaple (2)
$ ./glife ./sample_inputs/simple1_4_4 1 2 1 4 4
===============================
[0] *o*o
[1] *o*o
[2] o*oo
[3] oooo
===============================

:: Dump Row Column indices
0 0
0 2
1 0
1 2
2 1
Execution time(seconds): 0.000233
Program end...