#include <iostream>
#include <fstream>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstring>

#define LIVE 1
#define DEAD 0
#define DISPLAY 0
#define USECPSEC 1000000ULL

typedef struct _pthread_arg {
	int rows;
	int cols;
	int gen;
	int nprocs;
    long t_id;	
}pthread_arg;

class GameOfLifeGrid {
public:
	GameOfLifeGrid(int rows, int cols, int gen);
	
	int getNumOfNeighbors(int rows, int cols);
	
	int isLive(int rows, int cols) { return (m_Grid[rows][cols] ? LIVE : DEAD); }

	void dead(int rows, int cols) { m_Temp[rows][cols] = 0; }
	void live(int rows, int cols) { m_Temp[rows][cols] = 1; }

	void setCell(int rows, int cols) { m_Grid[rows][cols] = true; }
	
	void dump();
	void dumpIndex();

  	int** getGrid() { return m_Grid; }
	void updateGrid();

private:
	int** m_Grid;
	int** m_Temp;
	int m_Rows;
	int m_Cols;
	int m_Generations;

};

uint64_t dtime_usec(uint64_t start);