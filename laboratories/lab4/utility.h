#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct InputArgs
{
  long long start;   // Added start
  long long stop;    // Added stop
  int n_thr;
  char marker[64]; 
} Args;

void parseArgs(Args* aptr, int* argc, char** argv)
{
  // We now expect 3 arguments after the program name: start, stop, n_thr
  if (*argc < 4)
  {
    fprintf(stderr, "[Error] Too few arguments!\nUsage:\n%s [start] [stop] [n_thr]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  else
  {
    aptr->start = atoll(argv[1]);
    aptr->stop = atoll(argv[2]);
    aptr->n_thr = atoi(argv[3]);
    
    // Set a default marker so ins__printtime in main.c doesn't print garbage
    memset((aptr->marker), 0, 64);
    strncpy((aptr->marker), "OpenMP Sieve", 63); 
  }
  *argc = 1;
  return;
}

#ifdef CUDA
__host__
#endif
void ins__printtime(struct timeval *start, struct timeval *stop, char *marker) {

  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
  printf("\n%s: Execution time = %ld microseconds\n", marker, time);

  return;
}

#endif