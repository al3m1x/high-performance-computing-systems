#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

// Pakiet: 16 384 elementów w pamięci L1 (reprezentuje 32 768 liczb nieparzystych)
#define CHUNK_SIZE 16384 

__host__ void errorexit(const char *s) {
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

// ---------------------------------------------------------
// KROK 1: Inicjalizacja malutkiej tablicy bazowej (do sqrt(N))
// ---------------------------------------------------------
__global__ void init_base_sieve(char *base_sieve, long max_base_idx) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= max_base_idx) base_sieve[i] = 1;
}

// ---------------------------------------------------------
// KROK 2: Sito na tablicy bazowej 
// ---------------------------------------------------------
__global__ void run_base_sieve(char *base_sieve, long max_base_idx) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= max_base_idx) {
        long p = 2 * i + 3;
        if (p * p <= (2 * max_base_idx + 3)) {
            if (base_sieve[i] == 1) {
                long start_idx = (p * p - 3) / 2;
                for (long k = start_idx; k <= max_base_idx; k += p) {
                    base_sieve[k] = 0;
                }
            }
        }
    }
}

// ---------------------------------------------------------
// KROK 3: Segmentowane Sito Główne w pamięci __shared__
// ---------------------------------------------------------
__global__ void segmented_sieve(const char *base_sieve, long target_limit, long max_idx, long max_base_idx, unsigned long long *global_primes, unsigned long long *global_twins) {
    
    // Alokujemy CHUNK_SIZE + 1, żeby sprawdzanie pary siostrzanej na granicy pakietu nie wychodziło poza pamięć!
    __shared__ char s_sieve[CHUNK_SIZE + 1];
    __shared__ unsigned int block_primes;
    __shared__ unsigned int block_twins;

    if (threadIdx.x == 0) {
        block_primes = 0;
        block_twins = 0;
    }

    long chunk_start_idx = blockIdx.x * CHUNK_SIZE;
    long thread_idx = threadIdx.x;

    // 1. Inicjalizacja pakietu współdzielonego
    for (long i = thread_idx; i <= CHUNK_SIZE; i += blockDim.x) {
        s_sieve[i] = 1;
    }
    __syncthreads();

    long chunk_start_val = 2 * chunk_start_idx + 3;

    // 2. Równoległe wykreślanie w pakiecie z użyciem sita bazowego
    for (long i = thread_idx; i <= max_base_idx; i += blockDim.x) {
        if (base_sieve[i] == 1) {
            long p = 2 * i + 3;
            
            // Matematyka znajdująca pierwszą wielokrotność 'p' wpadającą w TEN konkretny pakiet
            long start_val = p * p;
            if (start_val < chunk_start_val) {
                long step = chunk_start_val / p;
                start_val = step * p;
                if (start_val < chunk_start_val) start_val += p;
                if (start_val % 2 == 0) start_val += p; // Musi być nieparzysta
            }

            long start_local_idx = (start_val - chunk_start_val) / 2;

            // Wykreślanie w ultraszybkiej pamięci lokalnej
            for (long j = start_local_idx; j <= CHUNK_SIZE; j += p) {
                s_sieve[j] = 0;
            }
        }
    }
    __syncthreads();

    // 3. Zliczanie w pakiecie
    unsigned int local_p = 0;
    unsigned int local_t = 0;

    for (long i = thread_idx; i < CHUNK_SIZE; i += blockDim.x) {
        long global_i = chunk_start_idx + i;
        
        if (global_i <= max_idx) {
            long val = 2 * global_i + 3;
            
            if (val <= target_limit) {
                if (s_sieve[i] == 1) {
                    local_p++;
                    
                    if (val + 2 <= target_limit) {
                        // Dzięki overlapowi CHUNK_SIZE + 1, indeks i + 1 zawsze leży w pamięci s_sieve
                        if (s_sieve[i + 1] == 1) {
                            local_t++;
                        }
                    }
                }
            }
        }
    }

    if (local_p > 0) atomicAdd(&block_primes, local_p);
    if (local_t > 0) atomicAdd(&block_twins, local_t);
    __syncthreads();

    // 4. Zrzut sum bloku do pamięci globalnej karty
    if (threadIdx.x == 0) {
        if (block_primes > 0) atomicAdd(global_primes, block_primes);
        if (block_twins > 0) atomicAdd(global_twins, block_twins);
    }
}

int main(int argc, char **argv) {
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);
    
    long target_limit = ins__args.arg; 

    long max_val = target_limit + 2; 
    long max_idx = (max_val - 3) / 2;
    if (max_idx < 0) max_idx = 0;

    // Obliczamy rozmiar dla małego Sita Bazowego do pierwiastka
    long sqrt_max = (long)sqrt((double)max_val);
    long max_base_idx = (sqrt_max - 3) / 2;
    if (max_base_idx < 0) max_base_idx = 0;

    unsigned long long h_primes = (target_limit >= 2) ? 1 : 0; 
    unsigned long long h_twins = 0;

    unsigned long long *d_primes = NULL;
    unsigned long long *d_twins = NULL;
    char *d_base_sieve = NULL;

    if (cudaSuccess != cudaMalloc((void **)&d_primes, sizeof(unsigned long long))) errorexit("Error allocating primes");
    if (cudaSuccess != cudaMalloc((void **)&d_twins, sizeof(unsigned long long))) errorexit("Error allocating twins");
    
    // Zauważ, jak mało pamięci alokujemy - max 4 kilobajty zamiast dawnych dziesiątek megabajtów!
    if (cudaSuccess != cudaMalloc((void **)&d_base_sieve, (max_base_idx + 1) * sizeof(char))) errorexit("Error allocating base sieve");

    if (cudaSuccess != cudaMemcpy(d_primes, &h_primes, sizeof(unsigned long long), cudaMemcpyHostToDevice)) errorexit("Error copying");
    if (cudaSuccess != cudaMemcpy(d_twins, &h_twins, sizeof(unsigned long long), cudaMemcpyHostToDevice)) errorexit("Error copying");

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);
    
    int threadsInBlock = 1024;

    // --- ETAP 1 & 2: Szybkie wygenerowanie bazy ---
    int blocksBaseInit = (max_base_idx + 1 + threadsInBlock - 1) / threadsInBlock;
    init_base_sieve<<<blocksBaseInit, threadsInBlock>>>(d_base_sieve, max_base_idx);
    cudaDeviceSynchronize();
    
    run_base_sieve<<<blocksBaseInit, threadsInBlock>>>(d_base_sieve, max_base_idx);
    cudaDeviceSynchronize();

    // --- ETAP 3: Zmasowany atak Segmentowanego Sita ---
    int blocksSegmented = (max_idx + CHUNK_SIZE) / CHUNK_SIZE; // Wystarczy zaledwie ok. 1500 bloków dla 50M
    segmented_sieve<<<blocksSegmented, threadsInBlock>>>(d_base_sieve, target_limit, max_idx, max_base_idx, d_primes, d_twins);
    cudaDeviceSynchronize();

    gettimeofday(&ins__tstop, NULL);

    if (cudaSuccess != cudaMemcpy(&h_primes, d_primes, sizeof(unsigned long long), cudaMemcpyDeviceToHost)) errorexit("Error copying");
    if (cudaSuccess != cudaMemcpy(&h_twins, d_twins, sizeof(unsigned long long), cudaMemcpyDeviceToHost)) errorexit("Error copying");

    printf("Przeszukano przedział od 1 do: %ld\n", target_limit);
    printf("Liczby pierwsze: %llu\n", h_primes);
    printf("Pary siostrzane: %llu\n", h_twins);

    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    cudaFree(d_primes);
    cudaFree(d_twins);
    cudaFree(d_base_sieve);

    return 0;
}