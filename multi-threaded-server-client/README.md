## **Multi-Threaded TCP Server**
- skeleton code : [Ted Baker](http://www.cs.fsu.edu/~baker/opsys/assign/P4.html)
- co-developer : [dmstmdrbs](https://github.com/dmstmdrbs)
 
### Description
* pthread API를 이용하여 여러 클라이언트와 연결되어 클라이언트들의 요청을 병렬처리하는 멀티 스레드 서버를 구현한 프로그램입니다.
* 서버와 클라이언트를 연결한 후, 클라이언트에 정수 N 을 입력하면 N 만큼의 랜덤 숫자를 만들어 하나씩 서버에게 전송합니다.
* 서버는 클라이언트가 보낸 숫자가 소수인지 아닌지 판별하여 그 결과를 클라이언트에게 전송합니다.
* 서버 실행시 만들어지는 스레드 개수를 직접 지정할 수 있습니다.
* 서버에 연결될 수 있는 최대 클라이언트 수는 서버에 만들어지는 스레드 개수의 1/3 으로 제한됩니다.

### Files
* multisrv.c : 멀티 스레드 서버가 구현되어 있는 c 코드입니다. 여러 클라이언트와 연결될 수 있고 클라이언트의 요청을 병렬처리 합니다.
* echocli.c : 클라이언트가 구현되어 있는 c 코드입니다. 클라이언트 역시 멀티 스레드로 구현되어 있으며, 서버에 요청을 보냄과 동시에 서버로부터 요청 처리 결과를 받을 수 있습니다.
* Makefile : 빌드 명령어를 포함하는 파일입니다.

### Usage
* 바이너리 파일 만들기 / 삭제하기
```shell
# How to Build?
$ make

$ make clean
```

* 바이너리 파일 실행하기
```shell
# Execute binary file
# Usage :
# [Server] ./multisrv -n <number of threads in server>
# <number of threads in server> must be greater than 2.
# [Client] SERVERHOST=localhost ./echocli <port number>

# Output; Server
$ ./multisrv -n 12
number of acceptors : 4, number of workers in pool : 8
server using port 42295

# Output; Client
$ SERVERHOST=localhost ./echocli 42295
connection wait time = 622 microseconds

$ 2
[SEND 0번째] :1804289383
[SEND 1번째] :1
[0번째] 1804289383 is not prime number
[1번째] 1 is not prime number
time : 1204492366
