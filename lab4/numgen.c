#include "numgen.h"
#include <stdlib.h>

#define SEED 12345678

unsigned int numgen(unsigned int count, unsigned long int dest[])
{

  unsigned int i = 0;

  srand(SEED);

  while(count--) {
    dest[i++] = rand();
  }

  return i;
}

