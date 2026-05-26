#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define DATA 0
#define RESULT 1
#define FINISH 2
#define MPI_RANGESIZE 1000000 // zwiększony zakres dla mpi do miliona
#define OMP_RANGESIZE 100000  // dla pojedynczego wątku w slave, openmp
#define QUEUE_SIZE 5

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


// funkcja obliczeniowa wykorzystująca wewnętrznie OpenMP
ResultMsg count_in_range(long long low_bound, long long high_bound, long long total_b, char *small_sieve, long long sqrt_n) {
    ResultMsg res = {0, 0};
    long long local_primes = 0;
    long long local_twins = 0;

    // Zrównoleglenie zadania przypisanego do danego slotu Slave'a
    #pragma omp parallel reduction(+:local_primes, local_twins)
    {
        char *mark = (char *)malloc((OMP_RANGESIZE + 2) * sizeof(char)); // teraz alokacja dla każdego wątku 

        // dynamiczny przydział sub-chunków do wątków
        #pragma omp for schedule(dynamic)
        for (long long low = low_bound; low <= high_bound; low += OMP_RANGESIZE) { // 100k na wątek
            
            long long high = low + OMP_RANGESIZE - 1;
            if (high > high_bound) high = high_bound;

            long long check_limit = (high + 2 > total_b) ? total_b : high + 2;
            long long size = check_limit - low + 1;

            for(long long i = 0; i < size; i++) {
                mark[i] = 0;
            }

            if (low <= 1) {
                if (low == 0) { mark[0] = 1; mark[1] = 1; }
                else if (low == 1) mark[0] = 1;
            }

            for (long long p = 2; p <= sqrt_n; p++) {
                if (!small_sieve[p]) {
                    long long start = (low <= p) ? p * p : ((low + p - 1) / p) * p;
                    for (long long j = start; j <= check_limit; j += p) {
                        mark[j - low] = 1;
                    }
                }
            }

            for (long long i = 0; i < (high - low + 1); i++) {
                long long current_num = low + i;
                
                if (!mark[i]) {
                    local_primes++;
                    
                    if (current_num + 2 <= total_b && !mark[i + 2]) {
                        local_twins++;
                    }
                }
            }
        }
        free(mark);
    }
    
    // zapis zsumowanych danych z wątków do wiadomości wyjściowej
    res.primes = local_primes;
    res.twins = local_twins;
    return res;
}

int main(int argc, char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);

    long long INITIAL_NUMBER = ins__args.start;
    long long FINAL_NUMBER = ins__args.stop;

    // ustawienie liczby wątków na podstawie argumentu programu
    omp_set_num_threads(ins__args.n_thr);

    int myrank, proccount;
    int threadsupport;

    // inicjalizacja MPI ze wsparciem dla wielowątkowości (FUNNELED = tylko główny wątek wywołuje MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threadsupport);
    
    if (threadsupport < MPI_THREAD_FUNNELED) {
        printf("Brak wsparcia dlaa MPI_THREAD_FUNNELED, rzeczywiste to %d\n", threadsupport);
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2) {
        if (myrank == 0) printf("Program wymaga co najmniej 2 procesów (1 master, co najmniej 1 slave).\n");
        MPI_Finalize();
        return 0;
    }
    
    int counter[proccount];
    for (int i=0; i<proccount; i++) {
        counter[i] = 0;
    }

    long long sqrt_n = (long long)sqrt(FINAL_NUMBER);
    char *small_sieve = (char *)calloc(sqrt_n + 1, sizeof(char));

    if (myrank == 0) {
        for (long long p = 2; p * p <= sqrt_n; p++)
            if (!small_sieve[p])
                for (long long i = p * p; i <= sqrt_n; i += p) small_sieve[i] = 1;
    }
    MPI_Bcast(small_sieve, sqrt_n + 1, MPI_CHAR, 0, MPI_COMM_WORLD); 

    if (myrank == 0) {
        // --- MASTER ---
        struct timeval ins__tstart, ins__tstop;
        gettimeofday(&ins__tstart, NULL);

        long long current_a = INITIAL_NUMBER;
        long long global_primes = 0, global_twins = 0;
        
        int num_slaves = proccount - 1;
        int total_slots = num_slaves * QUEUE_SIZE;
        
        TaskMsg *send_buffers = malloc(total_slots * sizeof(TaskMsg));
        MPI_Request *send_requests = malloc(total_slots * sizeof(MPI_Request));
        ResultMsg *recv_buffers = malloc(total_slots * sizeof(ResultMsg));
        MPI_Request *recv_requests = malloc(total_slots * sizeof(MPI_Request));

        for (int i = 0; i < total_slots; i++) {
            send_requests[i] = MPI_REQUEST_NULL;
            recv_requests[i] = MPI_REQUEST_NULL;
        }

        int active_requests = 0;

        for (int w = 0; w < num_slaves; w++) { 
            int slave_id = w + 1;
            for (int q = 0; q < QUEUE_SIZE; q++) { 
                int idx = w * QUEUE_SIZE + q; 
                if (current_a <= FINAL_NUMBER) {
                    send_buffers[idx].low = current_a;
                    // obliczanie dla dużych przedziałów w MPI
                    send_buffers[idx].high = (current_a + MPI_RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + MPI_RANGESIZE - 1;
                    
                    MPI_Isend(&send_buffers[idx], 2, MPI_LONG_LONG, slave_id, DATA, MPI_COMM_WORLD, &send_requests[idx]);
                    MPI_Irecv(&recv_buffers[idx], sizeof(ResultMsg), MPI_BYTE, slave_id, RESULT, MPI_COMM_WORLD, &recv_requests[idx]);
                    
                    current_a = send_buffers[idx].high + 1; 
                    active_requests++; 
                } else {
                    MPI_Isend(NULL, 0, MPI_LONG_LONG, slave_id, FINISH, MPI_COMM_WORLD, &send_requests[idx]);
                }
            }
        }

        while (active_requests > 0) {
            int idx;
            MPI_Status status;
            
            MPI_Waitany(total_slots, recv_requests, &idx, &status);
            
            global_primes += recv_buffers[idx].primes;
            global_twins += recv_buffers[idx].twins;
            active_requests--;

            MPI_Wait(&send_requests[idx], MPI_STATUS_IGNORE);

            int slave_id = (idx / QUEUE_SIZE) + 1; 

            if (current_a <= FINAL_NUMBER) {
                send_buffers[idx].low = current_a;
                send_buffers[idx].high = (current_a + MPI_RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + MPI_RANGESIZE - 1;
                
                MPI_Isend(&send_buffers[idx], 2, MPI_LONG_LONG, slave_id, DATA, MPI_COMM_WORLD, &send_requests[idx]);
                MPI_Irecv(&recv_buffers[idx], sizeof(ResultMsg), MPI_BYTE, slave_id, RESULT, MPI_COMM_WORLD, &recv_requests[idx]);
                
                current_a = send_buffers[idx].high + 1; 
                active_requests++;
            } else {
                MPI_Isend(NULL, 0, MPI_LONG_LONG, slave_id, FINISH, MPI_COMM_WORLD, &send_requests[idx]);
                recv_requests[idx] = MPI_REQUEST_NULL; 
            }
        }

        MPI_Waitall(total_slots, send_requests, MPI_STATUSES_IGNORE);

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
        // --- SLAVE ---
        TaskMsg tasks[QUEUE_SIZE];
        ResultMsg results[QUEUE_SIZE];
        MPI_Request recv_reqs[QUEUE_SIZE];
        MPI_Request send_reqs[QUEUE_SIZE];

        for (int i = 0; i < QUEUE_SIZE; i++) {
            send_reqs[i] = MPI_REQUEST_NULL;
            MPI_Irecv(&tasks[i], 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_reqs[i]);
        }

        int active_slots = QUEUE_SIZE;

        while (active_slots > 0) {
            int q;
            MPI_Status status;
            
            MPI_Waitany(QUEUE_SIZE, recv_reqs, &q, &status);

            if (status.MPI_TAG == FINISH) {
                active_slots--;
            } else {
                counter[myrank]++;
                // openmp wykonuje robotę ukrytą w count_in_range
                ResultMsg res = count_in_range(tasks[q].low, tasks[q].high, FINAL_NUMBER, small_sieve, sqrt_n);

                MPI_Wait(&send_reqs[q], MPI_STATUS_IGNORE);

                results[q] = res;
                MPI_Isend(&results[q], sizeof(ResultMsg), MPI_BYTE, 0, RESULT, MPI_COMM_WORLD, &send_reqs[q]);

                MPI_Irecv(&tasks[q], 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_reqs[q]);
            }
        }
        printf("Zakresy obsłużone w %d pakietach po stronie procesu [%d]: [%d]\n", MPI_RANGESIZE, myrank, counter[myrank]);
        MPI_Waitall(QUEUE_SIZE, send_reqs, MPI_STATUSES_IGNORE);
    }

    free(small_sieve);
    MPI_Finalize();
    return 0;
}