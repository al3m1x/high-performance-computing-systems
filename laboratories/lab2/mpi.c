#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define DATA 0
#define RESULT 1
#define FINISH 2
#define RANGESIZE 100000

// struktura do przedziału dla slave
typedef struct {
    long long low;
    long long high;
} TaskMsg;

// wynik dla mastera
typedef struct {
    long long primes;
    long long twins;
} ResultMsg;

// funkcja obliczeniowa do obliczania twins/primes
ResultMsg count_in_range(long long low_bound, long long high_bound, long long total_b, char *small_sieve, long long sqrt_n) {
    ResultMsg res = {0, 0};
    
    // liczby siostrzane na granicy
    long long check_limit = (high_bound + 2 > total_b) ? total_b : high_bound + 2;
    long long size = check_limit - low_bound + 1;

    char *mark = (char *)calloc(size, sizeof(char));
    if (low_bound <= 1) {
        if (low_bound == 0) { mark[0] = 1; mark[1] = 1; }
        else if (low_bound == 1) mark[0] = 1;
    }

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
            // siostrzane
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
        if (myrank == 0) printf("Program wymaga co najmniej 2 procesów (1 Master, 1+ Slave).\n");
        MPI_Finalize();
        return 0;
    }

    long long sqrt_n = (long long)sqrt(FINAL_NUMBER);
    char *small_sieve = (char *)calloc(sqrt_n + 1, sizeof(char));

    // małe sito do pierwiastka z n
    if (myrank == 0) {
        for (long long p = 2; p * p <= sqrt_n; p++)
            if (!small_sieve[p])
                for (long long i = p * p; i <= sqrt_n; i += p) small_sieve[i] = 1;
    }
    MPI_Bcast(small_sieve, sqrt_n + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        // MASTER
        struct timeval ins__tstart, ins__tstop;
        gettimeofday(&ins__tstart, NULL);

        long long current_a = INITIAL_NUMBER;
        long long global_primes = 0, global_twins = 0;
        int active_slaves = 0;
        
        // alokacja tablic dla komunikacji non blocking
        TaskMsg *send_buffers = malloc(proccount * sizeof(TaskMsg));
        MPI_Request *send_requests = malloc(proccount * sizeof(MPI_Request));
        ResultMsg *recv_buffers = malloc(proccount * sizeof(ResultMsg));
        MPI_Request *recv_requests = malloc(proccount * sizeof(MPI_Request));

        for (int i = 0; i < proccount; i++) {
            send_requests[i] = MPI_REQUEST_NULL;
            recv_requests[i] = MPI_REQUEST_NULL;
        }

        // rozdanie pierwszych zadań
        for (int i = 1; i < proccount; i++) {
            if (current_a <= FINAL_NUMBER) {
                send_buffers[i].low = current_a;
                send_buffers[i].high = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                
                // wysyłamy strukturę jako 2x long long
                MPI_Isend(&send_buffers[i], 2, MPI_LONG_LONG, i, DATA, MPI_COMM_WORLD, &send_requests[i]);
                MPI_Irecv(&recv_buffers[i], sizeof(ResultMsg), MPI_BYTE, i, RESULT, MPI_COMM_WORLD, &recv_requests[i]);
                
                current_a = send_buffers[i].high + 1;
                active_slaves++;
            }
        }

        // dynamiczne zbieranie wyników i przydzielanie nowych zadań
        while (active_slaves > 0) {
            int slave_id;
            MPI_Status status;
            
            // odbieramy wynik
            MPI_Waitany(proccount, recv_requests, &slave_id, &status);
            
            global_primes += recv_buffers[slave_id].primes;
            global_twins += recv_buffers[slave_id].twins;

            // upewniamy się, że poprzedni Isend do tego slave został sfinalizowany
            MPI_Wait(&send_requests[slave_id], MPI_STATUS_IGNORE);

            if (current_a <= FINAL_NUMBER) {
                send_buffers[slave_id].low = current_a;
                send_buffers[slave_id].high = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                
                MPI_Isend(&send_buffers[slave_id], 2, MPI_LONG_LONG, slave_id, DATA, MPI_COMM_WORLD, &send_requests[slave_id]);
                MPI_Irecv(&recv_buffers[slave_id], sizeof(ResultMsg), MPI_BYTE, slave_id, RESULT, MPI_COMM_WORLD, &recv_requests[slave_id]);
                current_a = send_buffers[slave_id].high + 1;
            } else {
                // brak zadań - wysyłamy sygnał końca pracy
                MPI_Isend(NULL, 0, MPI_LONG_LONG, slave_id, FINISH, MPI_COMM_WORLD, &send_requests[slave_id]);
                active_slaves--;
            }
        }

        // oczekiwanie na zakończenie wszystkich operacji
        if (proccount > 1) {
            MPI_Waitall(proccount - 1, &send_requests[1], MPI_STATUSES_IGNORE);
        }

        gettimeofday(&ins__tstop, NULL);
        printf("Zakres: [%lld, %lld]\n", INITIAL_NUMBER, FINAL_NUMBER);
        printf("Liczby pierwsze: %lld\n", global_primes);
        printf("Pary siostrzane: %lld\n", global_twins);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

        free(send_buffers);
        free(send_requests);
        free(recv_buffers);
        free(recv_requests);

    } else {
        // SLAVE
        TaskMsg current_task;
        TaskMsg next_task;
        ResultMsg res;
        MPI_Request req_next, req_res = MPI_REQUEST_NULL;
        MPI_Status status;

        // odbiór pierwszego zadania
        MPI_Recv(&current_task, 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        while (status.MPI_TAG != FINISH) {
            // rozpoczęcie nieblokującego pobierania kolejnego zadania (pre-fetching)
            MPI_Irecv(&next_task, 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &req_next);

            // obliczenia na bieżącym zakresie
            res = count_in_range(current_task.low, current_task.high, FINAL_NUMBER, small_sieve, sqrt_n);

            // tu jest wyścig bo res może się zmienić zanim Isend skończy wysylać niżej

            // wysyłanie wyniku nieblokująco
            if (req_res != MPI_REQUEST_NULL) MPI_Wait(&req_res, MPI_STATUS_IGNORE);
            MPI_Isend(&res, sizeof(ResultMsg), MPI_BYTE, 0, RESULT, MPI_COMM_WORLD, &req_res);

            // czekamy na zakończenie Irecv (pobranie kolejnych danych lub tagu FINISH)
            MPI_Wait(&req_next, &status);
            current_task = next_task;
        }
        
        // upewniamy się, że ostatni wynik wyszedł przed końcem procesu
        if (req_res != MPI_REQUEST_NULL) MPI_Wait(&req_res, MPI_STATUS_IGNORE);
    }

    free(small_sieve);
    MPI_Finalize();
    return 0;
}