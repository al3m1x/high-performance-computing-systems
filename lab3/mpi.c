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

    int N = 4; // rozmiar kolejki FIFO (paczki danych w obiegu na jednego slave'a)

    if (myrank == 0) {
        // MASTER
        struct timeval ins__tstart, ins__tstop;
        gettimeofday(&ins__tstart, NULL);

        long long current_a = INITIAL_NUMBER;
        long long global_primes = 0, global_twins = 0;
        
        int num_slaves = proccount - 1;
        int total_slots = num_slaves * N;
        
        // alokacja duzych tablic dla komunikacji non-blocking (buforowanie)
        TaskMsg *send_buffers = malloc(total_slots * sizeof(TaskMsg));
        MPI_Request *send_requests = malloc(total_slots * sizeof(MPI_Request));
        ResultMsg *recv_buffers = malloc(total_slots * sizeof(ResultMsg));
        MPI_Request *recv_requests = malloc(total_slots * sizeof(MPI_Request));

        for (int i = 0; i < total_slots; i++) {
            send_requests[i] = MPI_REQUEST_NULL;
            recv_requests[i] = MPI_REQUEST_NULL;
        }

        int active_requests = 0;

        // asynchroniczny initial batch - kazdy worker dostaje paczkę N zadań (każde z zadania ląduje w osobnym buforze okienka)
        for (int w = 0; w < num_slaves; w++) {
            int slave_id = w + 1;
            for (int q = 0; q < N; q++) {
                int idx = w * N + q; // plaski index 1D
                if (current_a <= FINAL_NUMBER) {
                    send_buffers[idx].low = current_a;
                    send_buffers[idx].high = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                    
                    // wysylamy asynchronicznie dane poczatkowe i od razu nasluchujemy na wynik w odpowiednim buforze
                    MPI_Isend(&send_buffers[idx], 2, MPI_LONG_LONG, slave_id, DATA, MPI_COMM_WORLD, &send_requests[idx]);
                    MPI_Irecv(&recv_buffers[idx], sizeof(ResultMsg), MPI_BYTE, slave_id, RESULT, MPI_COMM_WORLD, &recv_requests[idx]);
                    
                    current_a = send_buffers[idx].high + 1;
                    active_requests++;
                } else {
                    // brak poczatkowych zadan na zapelnienie okienek - od razu wysylamy sygnal zakonczenia tego slotu
                    MPI_Isend(NULL, 0, MPI_LONG_LONG, slave_id, FINISH, MPI_COMM_WORLD, &send_requests[idx]);
                }
            }
        }

        // pętla obsługująca odbiór wyników i rzucająca na bieżąco kolejne zadania
        while (active_requests > 0) {
            int idx;
            MPI_Status status;
            
            // Waitany od razu wyłapuje bufor ktory skonczył prace (daje nam to od ręki numer okienka 'idx')
            MPI_Waitany(total_slots, recv_requests, &idx, &status);
            
            global_primes += recv_buffers[idx].primes;
            global_twins += recv_buffers[idx].twins;
            active_requests--;

            // likwidacja wyscigu po stronie mastera: upewniamy się, że poprzedni wysył (Isend) dla tego okienka się zakończył zanim nadpiszemy taska
            MPI_Wait(&send_requests[idx], MPI_STATUS_IGNORE);

            int slave_id = (idx / N) + 1;

            if (current_a <= FINAL_NUMBER) {
                // mamy jeszcze co robic, wiec sypiemy zadan do zrodla ktore wlasnie zwolnilo okienko
                send_buffers[idx].low = current_a;
                send_buffers[idx].high = (current_a + RANGESIZE - 1 > FINAL_NUMBER) ? FINAL_NUMBER : current_a + RANGESIZE - 1;
                
                MPI_Isend(&send_buffers[idx], 2, MPI_LONG_LONG, slave_id, DATA, MPI_COMM_WORLD, &send_requests[idx]);
                MPI_Irecv(&recv_buffers[idx], sizeof(ResultMsg), MPI_BYTE, slave_id, RESULT, MPI_COMM_WORLD, &recv_requests[idx]);
                
                current_a = send_buffers[idx].high + 1;
                active_requests++;
            } else {
                // brak zadan - wysylamy tag FINISH aby ubic to okienko wewnatrz procesu slave
                MPI_Isend(NULL, 0, MPI_LONG_LONG, slave_id, FINISH, MPI_COMM_WORLD, &send_requests[idx]);
                recv_requests[idx] = MPI_REQUEST_NULL; // wyciszamy nasluch na ten slot
            }
        }

        // na koniec musimy miec pewnosc, ze wszystkie wyslane tagi FINISH faktycznie zostaly wypchniete do sieci
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
        // SLAVE
        
        // tworzymy bufory dla kolejki
        TaskMsg tasks[N];
        ResultMsg results[N];
        MPI_Request recv_reqs[N];
        MPI_Request send_reqs[N];

        for (int i = 0; i < N; i++) {
            send_reqs[i] = MPI_REQUEST_NULL;
            // pre-fetching: od razu wystawiamy Irecv na zapelnienie naszej wewnetrznej tuby FIFO
            MPI_Irecv(&tasks[i], 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_reqs[i]);
        }

        int active_slots = N;

        // dopóki mamy aktywne okienka w ktorych dzialamy
        while (active_slots > 0) {
            int q;
            MPI_Status status;
            
            // Waitany działa jak zjadanie z FIFO - bierze to, co pierwsze spadło z sieci do dowolnego z N buforów.
            // Jeśli żaden pakiet jeszcze nie dotarł, proces czeka (nie pali CPU w busy-waitingu).
            MPI_Waitany(N, recv_reqs, &q, &status);

            if (status.MPI_TAG == FINISH) {
                active_slots--;
                // UWAGA: MPI_Waitany automatycznie ustawia przetworzony recv_reqs[q] na MPI_REQUEST_NULL,
                // więc nasłuch na ten slot jest poprawnie wygaszany.
            } else {
                // obliczenia na bieżącym zakresie ze slotu
                ResultMsg res = count_in_range(tasks[q].low, tasks[q].high, FINAL_NUMBER, small_sieve, sqrt_n);

                // POZBYCIE SIE WYSCIGU!
                // czekamy z pewnoscia, ze poprzedni Isend dla TEGO SAMEGO okienka wyslal sie i zwolnil zasoby pamieci.
                MPI_Wait(&send_reqs[q], MPI_STATUS_IGNORE);

                // zapisujemy bezpiecznie nowy wynik do odseparowanego od innych iteracji bufora i rzucamy do mastera
                results[q] = res;
                MPI_Isend(&results[q], sizeof(ResultMsg), MPI_BYTE, 0, RESULT, MPI_COMM_WORLD, &send_reqs[q]);

                // wypychamy kolejne nasluchiwanie do rury (pre-fetching do wlasnie zuzytego zadania)
                MPI_Irecv(&tasks[q], 2, MPI_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_reqs[q]);
            }
        }
        
        // na sam koniec upewniamy się, że przed zabiciem procesu wszystkie wyniki zdazyly wyleciec rura po naszej stronie
        MPI_Waitall(N, send_reqs, MPI_STATUSES_IGNORE);
    }

    free(small_sieve);
    MPI_Finalize();
    return 0;
}