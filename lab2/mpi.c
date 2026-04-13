#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define DATA 0
#define RESULT 1
#define FINISH 2
#define RANGESIZE 100000 // wielkość zakresu dla każdego zadania (100k liczb)

typedef struct {
    long long primes;
    long long twins;
} ResultMsg;

// funkcja do obliczania liczby pierwszych i par siostrzanych w danym zakresie
ResultMsg count_in_range(long long low_bound, long long high_bound, long long total_b, char *small_sieve, long long sqrt_n) {
    ResultMsg res = {0, 0};
    // rozszerzamy zakres o 2, aby sprawdzić pary siostrzane na końcu bloku
    long long check_limit = (high_bound + 2 > total_b) ? total_b : high_bound + 2;
    long long size = check_limit - low_bound + 1;

    char *mark = (char *)calloc(size, sizeof(char));
    if (low_bound == 1) mark[0] = 1;

    for (long long p = 2; p <= sqrt_n; p++) {
        if (!small_sieve[p]) {
            long long start = (low_bound <= p) ? p * p : ((low_bound + p - 1) / p) * p;
            for (long long j = start; j <= check_limit; j += p) {
                mark[j - low_bound] = 1;
            }
        }
    }

    for (long long i = 0; i < (high_bound - low_bound + 1); i++) {
        long long current_num = low_bound + i;
        if (!mark[i]) {
            res.primes++;
            // sprawdzenie pary siostrzanej (p oraz p+2)
            if (current_num + 2 <= total_b && !mark[i + 2]) {
                res.twins++;
            }
        }
    }

    free(mark);
    return res;
}

int main(int argc, char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);

    long long INITIAL_NUMBER = ins__args.start;
    long long FINAL_NUMBER = ins__args.stop;

    int myrank, proccount;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2) {
        if (myrank == 0) printf("Uruchom z co najmniej 2 procesami (1 Master, 1+ Slave).\n");
        MPI_Finalize();
        return 0;
    }

    long long sqrt_n = (long long)sqrt(FINAL_NUMBER);
    char *small_sieve = (char *)calloc(sqrt_n + 1, sizeof(char));

    // 1. małe sito (wszyscy potrzebują wyników do pierwiastka z n)
    if (myrank == 0) {
        for (long long p = 2; p * p <= sqrt_n; p++)
            if (!small_sieve[p])
                for (long long i = p * p; i <= sqrt_n; i += p) small_sieve[i] = 1;
    }
    MPI_Bcast(small_sieve, sqrt_n + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // broadcast do wszystkich procesów

    if (myrank == 0) {
        // logika mastera (Dynamic Load Balancing)
        struct timeval ins__tstart, ins__tstop;
        gettimeofday(&ins__tstart, NULL);

        long long current_a = INITIAL_NUMBER;
        long long global_primes = 0, global_twins = 0;
        int active_slaves = 0;
        long long range[2];
        MPI_Status status;

        // rozdaje pierwsze zadania wszystkim niewolnikom
        for (int i = 1; i < proccount; i++) {
            if (current_a <= FINAL_NUMBER) {
                range[0] = current_a;
                range[1] = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                
                MPI_Send(range, 2, MPI_LONG_LONG, i, DATA, MPI_COMM_WORLD);
                current_a = range[1] + 1;
                active_slaves++;
            }
        }

        // zbieranie wyników i wysyłanie nowych zadań (Dynamic Load Balancing)
        while (active_slaves > 0) {
            ResultMsg partial_res;
            MPI_Recv(&partial_res, sizeof(ResultMsg), MPI_BYTE, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            global_primes += partial_res.primes;
            global_twins += partial_res.twins;

            if (current_a <= FINAL_NUMBER) {
                range[0] = current_a;
                range[1] = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                
                MPI_Send(range, 2, MPI_LONG_LONG, status.MPI_SOURCE, DATA, MPI_COMM_WORLD);
                current_a = range[1] + 1;
            } else {
                // koniec pracy dla niewolnika
                MPI_Send(NULL, 0, MPI_LONG_LONG, status.MPI_SOURCE, FINISH, MPI_COMM_WORLD);
                active_slaves--;
            }
        }

        gettimeofday(&ins__tstop, NULL);
        printf("Zakres: %lld - %lld\n", INITIAL_NUMBER, FINAL_NUMBER);
        printf("Liczby pierwsze: %lld\n", global_primes);
        printf("Pary siostrzane: %lld\n", global_twins);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    } else {
        // logika niewolnika (Dynamic Load Balancing)
        long long range_current[2];
        long long range_next[2];
        MPI_Request req_next;
        MPI_Status status;

        // odbierz pierwsze zadanie blokująco
        MPI_Recv(range_current, 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        while (status.MPI_TAG != FINISH) {
            // pre-fetching następnego zadania (nieblokująco)
            MPI_Irecv(range_next, 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req_next);

            // obliczanie
            ResultMsg res = count_in_range(range_current[0], range_current[1], FINAL_NUMBER, small_sieve, sqrt_n);

            // nieblokujące wysłanie wyniku do mastera
            MPI_Request req_res;
            MPI_Isend(&res, sizeof(ResultMsg), MPI_BYTE, 0, RESULT, MPI_COMM_WORLD, &req_res);

            // czekaj na zakończenie Irecv (następne dane)
            MPI_Wait(&req_next, &status);
            
            // przesunięcie zakresu do aktualnego
            range_current[0] = range_next[0];
            range_current[1] = range_next[1];
            
            // czekaj na zakończenie Isend (wynik)
            MPI_Wait(&req_res, MPI_STATUS_IGNORE);
        }
    }

    free(small_sieve);
    MPI_Finalize();
    return 0;
}