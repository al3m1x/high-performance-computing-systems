#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>

// Usunięto numgen.c

#define RANGESIZE 100000

int main(int argc, char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  // set number of threads
  omp_set_num_threads(ins__args.n_thr);

  // Pobranie granic przedziału (analogicznie do Twojego kodu MPI)
  long long INITIAL_NUMBER = ins__args.start;
  long long FINAL_NUMBER = ins__args.stop;

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  // -------------------------------------------------------------------
  // RUN YOUR COMPUTATIONS HERE (INCLUDING OPENMP STUFF)
  // -------------------------------------------------------------------

  long long prime_count = 0;
  long long twin_prime_pairs = 0;

  // 1. Sekwencyjne małe sito do sqrt(FINAL_NUMBER)
  long long sqrt_n = (long long)sqrt(FINAL_NUMBER);
  char *small_sieve = (char *)calloc(sqrt_n + 1, sizeof(char));

  for (long long p = 2; p * p <= sqrt_n; p++) {
      if (!small_sieve[p]) {
          for (long long i = p * p; i <= sqrt_n; i += p) {
              small_sieve[i] = 1;
          }
      }
  }

  // 2. Równoległe przetwarzanie porcjami (Segmented Sieve)
  #pragma omp parallel reduction(+:prime_count, twin_prime_pairs)
  {
      // Prywatna tablica dla każdego wątku (+2 elementy na zakładkę dla liczb siostrzanych na krawędzi bloku)
      char *mark = (char *)malloc((RANGESIZE + 2) * sizeof(char));

      // Dynamiczny przydział bloków (chunków) do wątków
      #pragma omp for schedule(dynamic)
      for (long long low = INITIAL_NUMBER; low <= FINAL_NUMBER; low += RANGESIZE) {
          
          long long high = low + RANGESIZE - 1;
          if (high > FINAL_NUMBER) high = FINAL_NUMBER;

          // Limit sprawdzania rozszerzony o 2, by łatwo wyłapać siostrzane na granicy
          long long check_limit = (high + 2 > FINAL_NUMBER) ? FINAL_NUMBER : high + 2;
          long long size = check_limit - low + 1;

          // Wyzerowanie prywatnej tablicy mark
          for(long long i = 0; i < size; i++) {
              mark[i] = 0;
          }

          // Krawędziowy przypadek dla 0 i 1
          if (low <= 1) {
              if (low == 0) { mark[0] = 1; mark[1] = 1; }
              else if (low == 1) mark[0] = 1;
          }

          // Wykreślanie wielokrotności w obecnym małym przedziale
          for (long long p = 2; p <= sqrt_n; p++) {
              if (!small_sieve[p]) {
                  long long start = (low <= p) ? p * p : ((low + p - 1) / p) * p;
                  for (long long j = start; j <= check_limit; j += p) {
                      mark[j - low] = 1;
                  }
              }
          }

          // Zliczanie w tym przedziale
          for (long long i = 0; i < (high - low + 1); i++) {
              long long current_num = low + i;
              
              if (!mark[i]) {
                  prime_count++;
                  
                  // Sprawdzenie pary siostrzanej
                  if (current_num + 2 <= FINAL_NUMBER && !mark[i + 2]) {
                      twin_prime_pairs++;
                  }
              }
          }
      }

      // Zwolnienie prywatnej pamięci wątku na koniec sekcji równoległej
      free(mark);
  }

  free(small_sieve);

  // Print results
  printf("Zakres: [%lld, %lld]\n", INITIAL_NUMBER, FINAL_NUMBER);
  printf("Total Primes Found: %lld\n", prime_count);
  printf("Total Twin Prime Pairs: %lld\n", twin_prime_pairs);

  // -------------------------------------------------------------------
  // END OF COMPUTATIONS
  // -------------------------------------------------------------------

  // synchronize/finalize your computations
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

  return 0;
}