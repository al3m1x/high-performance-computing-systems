#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "numgen.c"

int main(int argc, char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // set number of threads
  omp_set_num_threads(ins__args.n_thr);

  // program input argument
  long inputArgument = ins__args.arg; 
  unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  // -------------------------------------------------------------------
  // RUN YOUR COMPUTATIONS HERE (INCLUDING OPENMP STUFF)
  // -------------------------------------------------------------------

  // 1. Find the maximum value in the generated numbers to determine Sieve size
  unsigned long int max_val = 0;

  #pragma omp parallel for reduction(max:max_val)
  for (long i = 0; i < inputArgument; i++) {
      if (numbers[i] > max_val) {
          max_val = numbers[i];
      }
  }

  // 2. Allocate Sieve array (up to max_val + 2 to safely check twin primes)
  unsigned char *sieve = (unsigned char *)malloc((max_val + 3) * sizeof(unsigned char));
  if (sieve == NULL) {
      fprintf(stderr, "Memory allocation failed for Sieve. Numbers are too large.\n");
      free(numbers);
      return 1;
  }

  // Initialize the sieve array to 1 (true)
  #pragma omp parallel for
  for (unsigned long int i = 0; i <= max_val + 2; i++) {
      sieve[i] = 1;
  }
  sieve[0] = 0;
  sieve[1] = 0;

  // 3. Sieve of Eratosthenes (Parallelized)
  // Outer loop is sequential, inner loop (crossing out multiples) is highly parallel
  for (unsigned long int p = 2; p * p <= max_val + 2; p++) {
      if (sieve[p]) {
          #pragma omp parallel for schedule(auto)
          for (unsigned long int i = p * p; i <= max_val + 2; i += p) {
              sieve[i] = 0;
          }
      }
  }

  // 4. Count Primes and Twin Primes in the numbers array
  long prime_count = 0;
  long twin_prime_pairs = 0;

  #pragma omp parallel for reduction(+:prime_count, twin_prime_pairs)
  for (long i = 0; i < inputArgument; i++) {
      unsigned long int num = numbers[i];

      if (sieve[num]) {
          prime_count++;

          // Count a twin prime pair if (num, num+2) are both prime.
          // Because we built the sieve up to max_val + 2, this is memory-safe.
          if (sieve[num + 2]) {
              twin_prime_pairs++;
          }
      }
  }

  // Print results
  printf("Total Primes Found: %ld\n", prime_count);
  printf("Total Twin Prime Pairs: %ld\n", twin_prime_pairs);

  // Free memory to prevent leaks
  free(sieve);
  free(numbers);

  // -------------------------------------------------------------------
  // END OF COMPUTATIONS
  // -------------------------------------------------------------------

  // synchronize/finalize your computations
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

  return 0;
}