#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

void sieve_primes_and_twins(long long a, long long b, int myrank, int proccount) {
    long long n = b;
    long long sqrt_n = (long long)sqrt(n);
    long long n_per_proc = n / proccount;
    long long low_bound = myrank * n_per_proc + 1;
    long long high_bound = (myrank == proccount - 1) ? n : (myrank + 1) * n_per_proc;
    long long size = high_bound - low_bound + 1;

    // tablica lokalna: 0 = l. pierwsza, 1 = l. wykreślona
    char *mark = (char *)calloc(size, sizeof(char));
    if (low_bound == 1) mark[0] = 1; // 1 nie jest pierwsza

    // małe sito na pierwszym procesie do znalezienia liczb pierwszych do sqrt(n)
    char *small_sieve = (char *)calloc(sqrt_n + 1, sizeof(char));
    if (myrank == 0) { // tylko jeden proces, bo pierwiastek sprawia że nie opłaca raczej się tego dzielić (chyba)
        for (long long p = 2; p * p <= sqrt_n; p++)
            if (!small_sieve[p])
                for (long long i = p * p; i <= sqrt_n; i += p) small_sieve[i] = 1;
    }
    MPI_Bcast(small_sieve, sqrt_n + 1, MPI_CHAR, 0, MPI_COMM_WORLD); // broadcast małego sita do wszystkich procesów

    // wykreślanie lokalne
    for (long long p = 2; p <= sqrt_n; p++) {
        if (!small_sieve[p]) {
            long long start = (low_bound <= p) ? p * p : ((low_bound + p - 1) / p) * p; // początek
            for (long long j = start; j <= high_bound; j += p) mark[j - low_bound] = 1; // wykreślamy wielokrotności p w zakresie
        }
    }

    // zliczenie lokalnych pierwszych i siostrzanych (wewnątrz bloku)
    long long local_primes = 0, local_twins = 0;
    for (long long i = 0; i < size; i++) {
        long long current_num = low_bound + i;
        // zliczamy tylko jeśli liczba jest w zadanym zakresie a do b
        if (current_num >= a && !mark[i]) {
            local_primes++;
            if (i + 2 < size && !mark[i + 2] && (current_num + 2 <= b)) local_twins++;
        }
    }

    // pary siostrzane na stykach bloków
    // sprawdzamy dwie ostatnie liczby bloku, bo mogą tworzyć parę z dwiema pierwszymi sąsiada
    char neighbor_start[2] = {1, 1}; // domyślnie niepierwsze
    if (myrank < proccount - 1) {
        MPI_Recv(neighbor_start, 2, MPI_CHAR, myrank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // sprawdzamy styk tylko jeśli liczby mieszczą się w zakresie a do b
        long long num_end = high_bound;
        if (num_end >= a && num_end + 2 <= b) {
            // jeśli moja ostatnia liczba (index size-1) jest pierwsza i sąsiada druga (index 1) jest pierwsza
            if (!mark[size - 1] && !neighbor_start[1]) local_twins++;
            // jeśli moja przedostatnia (index size-2) i sąsiada pierwsza (index 0) są pierwsze
            if (num_end - 1 >= a && !mark[size - 2] && !neighbor_start[0]) local_twins++;
        }
    }
    if (myrank > 0) {
        char my_start[2] = {mark[0], (size > 1 ? mark[1] : (char)1)};
        MPI_Send(my_start, 2, MPI_CHAR, myrank - 1, 0, MPI_COMM_WORLD);
    }

    long long global_primes, global_twins;
    MPI_Reduce(&local_primes, &global_primes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // sumujemy liczby pierwsze
    MPI_Reduce(&local_twins, &global_twins, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // sumujemy liczby siostrzane

    if (myrank == 0) {
        printf("Zakres: %lld - %lld\n", a, b);
        printf("Liczby pierwsze: %lld\n", global_primes);
        printf("Pary siostrzane: %lld\n", global_twins);
    }
    free(mark); free(small_sieve);
}


int main(int argc, char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);

    long long INITIAL_NUMBER = ins__args.start; 
    long long FINAL_NUMBER = ins__args.stop;

    struct timeval ins__tstart, ins__tstop;
    int myrank, nproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // pomiar czasu zaczynamy tylko na procesie 0
    if (!myrank) 
        gettimeofday(&ins__tstart, NULL);

    // wywołanie sita
    sieve_primes_and_twins(INITIAL_NUMBER, FINAL_NUMBER, myrank, nproc);

    // synchronizacja i wypisanie czasu na końcu
    if (!myrank) {
        gettimeofday(&ins__tstop, NULL);
        ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
    }

    MPI_Finalize();
    return 0;
}
