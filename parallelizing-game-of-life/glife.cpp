#include "glife.h"

using namespace std;

int gameOfLife(int argc, char *argv[]);
void singleThread(int, int, int);
void* workerThread(void *);

int nprocs;
pthread_barrier_t barrier;
GameOfLifeGrid* g_GameOfLifeGrid;

uint64_t dtime_usec(uint64_t start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

GameOfLifeGrid::GameOfLifeGrid(int rows, int cols, int gen)
{
  m_Generations = gen;
  m_Rows = rows;
  m_Cols = cols;

  m_Grid = (int**)malloc(sizeof(int*) * rows);
  if (m_Grid == NULL) 
    cout << "1 Memory allocation error " << endl;

  m_Temp = (int**)malloc(sizeof(int*) * rows);
  if (m_Temp == NULL) 
    cout << "2 Memory allocation error " << endl;

  m_Grid[0] = (int*)malloc(sizeof(int) * (cols*rows));
  if (m_Grid[0] == NULL) 
    cout << "3 Memory allocation error " << endl;

  m_Temp[0] = (int*)malloc(sizeof(int) * (cols*rows));	
  if (m_Temp[0] == NULL) 
    cout << "4 Memory allocation error " << endl;

  for (int i = 1; i < rows; i++) {
    m_Grid[i] = m_Grid[i-1] + cols;
    m_Temp[i] = m_Temp[i-1] + cols;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m_Grid[i][j] = m_Temp[i][j] = 0;
    }
  }
}

// Entry point
int main(int argc, char* argv[])
{
  if (argc != 7) {
    cout <<"Usage: " << argv[0] << " <input file> <display> <nprocs>"
           " <# of generation> <width> <height>" << endl;
    cout <<"\n\tnprocs = 0: Running sequentiallyU" << endl;
    cout <<"\tnprocs =1: Running on a single thread" << endl;
    cout <<"\tnprocs >1: Running on multiple threads" << endl;
    cout <<"\tdisplay = 1: Dump results" << endl;
    return 1;
  }

  return gameOfLife(argc, argv);
}

int gameOfLife(int argc, char* argv[])
{
  int cols, rows, gen;
  ifstream inputFile;
  int input_row, input_col, display;
  uint64_t difft;
  pthread_t *threadID;

  inputFile.open(argv[1], ifstream::in);

  if (inputFile.is_open() == false) {
    cout << "The "<< argv[1] << " file can not be opend" << endl;
    return 1;
  }

  display = atoi(argv[2]);
  nprocs = atoi(argv[3]);
  gen = atoi(argv[4]);
  cols = atoi(argv[5]);
  rows = atoi(argv[6]);

  g_GameOfLifeGrid = new GameOfLifeGrid(rows, cols, gen);

  while (inputFile.good()) {
    inputFile >> input_row >> input_col;
    if (input_row >= rows || input_col >= cols) {
      cout << "Invalid grid number" << endl;
      return 1;
    }else
      g_GameOfLifeGrid->setCell(input_row, input_col);
  }

  if(rows < nprocs){
    cout << "Invalid nprocs. Number of threads must be less than the grid height." << endl;
    return 1;
  }

  // Start measuring execution time
  difft = dtime_usec(0);

  int neighbors = 0;
  if (nprocs == 0) {
    //////////////////////////////////////////
    // Running with sequential version //  
    int neighbors = 0;
    for(int g=0; g<gen; g++){
      for(int r=0; r<rows; r++){
        for(int c=0; c<cols; c++){

          neighbors = g_GameOfLifeGrid->getNumOfNeighbors(r, c);
          if(neighbors < 2 || neighbors > 3) g_GameOfLifeGrid->dead(r, c);
          else if(neighbors == 3 || (g_GameOfLifeGrid->isLive(r,c) && neighbors == 2)) g_GameOfLifeGrid->live(r, c);
          else g_GameOfLifeGrid->dead(r, c);

        } // end of for c
      } // end of for r
      g_GameOfLifeGrid->updateGrid();
    } // end of for g
    //////////////////////////////////////////

  } else if (nprocs == 1) {
    /////////////////////////////
    // Running a single thread //
    singleThread(rows, cols, gen);
    /////////////////////////////

  } else { 
    ////////////////////////////////////////
    // Running multiple threads (pthread) //
    pthread_t threads[nprocs];
    int rc;
    long t;
    long status;

    // initialize barrior
    rc = pthread_barrier_init(&barrier, NULL, nprocs);
    if(rc){
	    cout << "Error; return code from pthread_barrier_init() is " << rc <<  endl;
	    exit(-1);
    }

    // create the threads
    for(t = 0; t<nprocs; t++){
	    pthread_arg *t_arg = (pthread_arg*)malloc(sizeof(pthread_arg));
	    t_arg->rows = rows;
	    t_arg->cols = cols;
	    t_arg->gen = gen;
	    t_arg->nprocs = nprocs;
	    t_arg->t_id = t;

	    rc = pthread_create(&threads[t], NULL, workerThread, (void*)t_arg);
	    if(rc){
	      cout << "Error; return code from pthread_create() is " << rc << endl;
      	exit(-1);
      }	    
    }

    // waits for the execution of all threads
    for(t=0; t<nprocs; t++){
	    rc = pthread_join(threads[t], (void**)&status);
      if (rc) {
		    printf("Error; return code from pthread_join() is %d\n", rc);
		    exit(-1);
	    }
    }

    // destory pthread barrior
    pthread_barrier_destroy(&barrier);
    ///////////////////////////////////////

  } // end of if-else nprocs

  // End measuring execution time
  difft = dtime_usec(difft);

  // Print indices
  if (display && nprocs){
    g_GameOfLifeGrid->dump();
    g_GameOfLifeGrid->dumpIndex();
  }

  // Print execution time
  if (nprocs == 0) {
    // Sequential version
    cout << "Execution time(seconds): " << difft/(float)USECPSEC << endl;
  } else if (nprocs >= 1) {
    // Single or multi-thread execution time 
    cout << "Execution time(seconds): " << difft/(float)USECPSEC << endl;
  }
  inputFile.close();
  cout << "Program end... " << endl;

  return 0;
} // end of main

void singleThread(int rows, int cols, int gen)
{	
	int neighbors = 0;
	for(int g=0; g<gen; g++){
    for(int r=0; r<rows; r++){
      for(int c=0; c<cols; c++){

        // determine the state of the cell (alive or dead)
        neighbors = g_GameOfLifeGrid->getNumOfNeighbors(r, c);
        if(neighbors < 2 || neighbors > 3) g_GameOfLifeGrid->dead(r, c);
        else if(neighbors == 3 || (g_GameOfLifeGrid->isLive(r,c) && neighbors == 2)) g_GameOfLifeGrid->live(r, c);
        else g_GameOfLifeGrid->dead(r, c);

      } // end of for c
    } // end of for r

    // update grid
    g_GameOfLifeGrid->updateGrid();

  } // end of for g
}

void* workerThread(void *arg)
{
	pthread_arg *t_arg = (pthread_arg*)arg;
	int t_area = t_arg->rows / t_arg->nprocs;
	int t_start_row = t_arg->t_id * t_area;
	int t_end_row = (t_arg->t_id == t_arg->nprocs - 1) ? t_arg->rows : t_start_row + t_area;
	int rc;

	int neighbors = 0;
  for(int g=0; g<t_arg->gen; g++){
    for(int r=t_start_row; r<t_end_row; r++){
      for(int c=0; c<t_arg->cols; c++){

        // determine the state of the cell (alive or dead)
        neighbors = g_GameOfLifeGrid->getNumOfNeighbors(r, c);
        if(neighbors < 2 || neighbors > 3) g_GameOfLifeGrid->dead(r, c);
        else if(neighbors == 3 || (g_GameOfLifeGrid->isLive(r,c) && neighbors == 2)) g_GameOfLifeGrid->live(r, c);
        else g_GameOfLifeGrid->dead(r, c);

      } // end of for c
    } // end of for r
		
    //////////////////////////
		// pthread barrior wait //
		rc = pthread_barrier_wait(&barrier);
		if (rc == EINVAL) {
			cout << "Error; return code from pthread_barrier_wait() is " << rc <<  endl;
      exit(-1);
    }

    // update grid
    if(t_arg->t_id == 0) g_GameOfLifeGrid->updateGrid();

    // pthread barrior wait
		rc = pthread_barrier_wait(&barrier);
    if (rc == EINVAL) {
      cout << "Error; return code from pthread_barrier_wait() is " << rc <<  endl;
      exit(-1);
    }
    //////////////////////////

  } // end of for g
}

int GameOfLifeGrid::getNumOfNeighbors(int rows, int cols)
{
  int numOfNeighbors = 0;

  for(int i=-1; i<2; i++){
    for(int j=-1; j<2; j++){

      if((i==0 && j==0) || (rows+i<0) || (cols+j<0) || (rows+i>=m_Rows) || (cols+j>=m_Cols)) continue;
      else if (m_Grid[rows+i][cols+j] == true) numOfNeighbors++;

    }
  }

  return numOfNeighbors;
}

void GameOfLifeGrid::updateGrid(){
	copy(&m_Temp[0][0], &m_Temp[0][0] + m_Rows * m_Cols, &m_Grid[0][0]);
}

void GameOfLifeGrid::dump() 
{
  cout << "===============================" << endl;

  for (int i = 0; i < m_Rows; i++) {
    cout << "[" << i << "] ";
    for (int j = 0; j < m_Cols; j++) {
      if (m_Grid[i][j] == 1)
        cout << "*";
      else
        cout << "o";
    }
    cout << endl;
  }
  cout << "===============================\n" << endl;
}

void GameOfLifeGrid::dumpIndex()
{
  cout << ":: Dump Row Column indices" << endl;
  for (int i=0; i < m_Rows; i++) {
    for (int j=0; j < m_Cols; j++) {
      if (m_Grid[i][j]) cout << i << " " << j << endl;
    }
  }
}