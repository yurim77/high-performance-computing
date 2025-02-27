#define DEQUE_LEN 5
#define TRUE 1
#define FALSE 0

#include <stdlib.h>
#include <stdio.h>

typedef struct _pthread_arg {
  void (*taskFunction)(float* , float* , float *, int , int , int , int , int, int, int); /* function pointer */
  int ARRAY_HEIGHT;
  int ARRAY_WIDTH;
  int PADD_RADIUS;
  int KERNEL_SIZE;
  // int depth;
  int start_depth;
  int end_depth;
  float* cache_in;
  float* cache_out;
  float* kernel;
  // float** lines;
  long t_id;
} pthread_arg;

typedef struct Task{
  void (*taskFunction)(float*, float*, float *, int,  int ,  int ,  int , int , int ); /* function pointer */
  int ARRAY_HEIGHT;
  int ARRAY_WIDTH;
  int PADD_RADIUS;
  int KERNEL_SIZE;
  // int depth;
  int start_depth;
  int end_depth;
  float* cache_in;
  float* cache_out;
  float* kernel;
  // float** lines;
  long t_id;
} Task;

typedef struct TaskDeque{
  Task* tasks;
  int rear;
  int front;
}TaskDeque;

void init_deque(TaskDeque *q);
int isEmpty(TaskDeque *q);
int isFull(TaskDeque *q);
void add_front(TaskDeque *q, Task task);
void add_rear(TaskDeque *q, Task task);
Task get_front(TaskDeque *q);
Task get_rear(TaskDeque *q);
Task pop_front(TaskDeque *q);
Task pop_rear(TaskDeque *q);

void init_deque(TaskDeque *q) {
    q->tasks = (Task*)malloc(sizeof(Task)*DEQUE_LEN);
    q->front = 0;
    q->rear = 0;
}

int isEmpty(TaskDeque *q){
    if(q->front == q->rear) return TRUE;
    else return FALSE;
}

int isFull(TaskDeque *q){
  if(((q->rear+1)%DEQUE_LEN)==q->front) return TRUE;
  else return FALSE;
}

void add_rear(TaskDeque *q, Task task){
  if(isFull(q)){printf("Deque is Full\n"); return;}
  else{
    q->rear = (q->rear+1) % DEQUE_LEN;
    q->tasks[q->rear]=task;
  }
}

void add_front(TaskDeque *q, Task task){
  if(isFull(q)) { printf("Deque is Full\n"); return; }
  else {
    q->tasks[q->front] = task;
    q->front = (q->front -1 + DEQUE_LEN) % DEQUE_LEN;
    return;
  }
}


Task get_front(TaskDeque *q){
  if(isEmpty(q)) printf("Deque is Empty\n");
  else return q->tasks[(q->front+1)%DEQUE_LEN];
}

Task get_rear(TaskDeque *q){
  if(isEmpty(q)) {
    printf("Deque is Empty\n");
  }else{
    return q->tasks[q->rear];
  }
}

Task pop_front(TaskDeque *q){
  if(isEmpty(q)) {
    printf("Deque is Empty\n");
  }else{
    Task poped = get_front(q);
    q->front = (q->front + 1) % DEQUE_LEN;
    return poped;
  }
}

Task pop_rear(TaskDeque *q) {
  if(isEmpty(q)) {
    printf("Deque is Empty\n");
  }else{
    Task poped = get_rear(q);
    q->rear = (q->rear - 1 + DEQUE_LEN) % DEQUE_LEN;
    return poped;
  }
}