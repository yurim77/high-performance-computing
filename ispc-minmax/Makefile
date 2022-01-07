export PATH := $(PATH):$(HOME)/ispc-v1.16.1-linux/bin

all: main.cpp minmax.ispc
	@export PATH
	@ispc minmax.ispc -o minmax_ispc.o
	@ispc minmax.ispc -h minmax_ispc.h
	@g++ -c main.cpp -o main.o
	@g++ main.o minmax_ispc.o -o minmax
	@echo successfully completed.

clean: 
	@rm main.o minmax_ispc.o minmax_ispc.h minmax
	@echo successfully completed.
