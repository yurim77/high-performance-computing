all: glife

glife: glife.o
	@g++ -o glife glife.o -pthread

glife.o: glife.h glife.cpp
	@g++ -c -o glife.o glife.cpp -pthread

clean: 
	@rm -f glife *.o
