## **ISPC Programming**

### Description
* 인텔 ISPC (Implicit SPMD Program Compiler) 를 이용하여 배열의 최댓값과 최솟값을 찾는 프로그램입니다.

### Environment
* linux 5.15.4
* 홈 디렉토리에 pre-compile 된 ISPC 소스 코드와 바이너리가 있어야 합니다.
* 아래 링크에서 Linux (64 bit) ispc binary and examples (v1.16.1) 을 다운받으세요.
* https://ispc.github.io/downloads.html
* 또는, 아래 명령어를 이용할 수 있습니다.

```shell
# download
$wget https://github.com/ispc/ispc/releases/download/v1.16.1/ispc-v1.16.1-linux.tar.gz
```

---

### Files
* main.cpp : 메인 함수가 정의되어있는 c++ 코드입니다.
* minmax.ispc : 배열의 최댓값과 최솟값을 찾는 함수가 정의되어있는 ispc 코드입니다.
* Makefile : 빌드 명령어를 포함하는 파일입니다.

---

### Usage
* 바이너리 파일 만들기 / 삭제하기
```shell
# How to Build?
$ make all
Warning: No --target specified on command-line. Using default
        system target "avx2-i32x8".
Warning: No --target specified on command-line. Using default
        system target "avx2-i32x8".
successfully completed.

$ make clean
successfully completed.
```

* 바이너리 파일 실행하기
```shell
# Execute binary file
# Output
$ ./minmax
min value is 0
max value is 1023