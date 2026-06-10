#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define CHUNK_SIZE 16384 

__host__ void errorexit(const char *s) {
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__global__ void init_base_sieve(char *base_sieve, long max_base_val) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= max_base_val) {
        // z góry wiemy, że 0 i 1 nie są pierwsze
        base_sieve[i] = (i >= 2) ? 1 : 0;
    }
}

__global__ void run_base_sieve(char *base_sieve, long max_base_val) {
    long p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p >= 2 && p <= max_base_val) {
        if (p * p <= max_base_val) {
            if (base_sieve[p] == 1) {
                // małe sito - wykreślanie wielokrotności
                for (long k = p * p; k <= max_base_val; k += p) {
                    base_sieve[k] = 0;
                }
            }
        }
    }
}

__global__ void segmented_sieve(const char *base_sieve, long target_limit, long max_base_val, unsigned long long *global_primes, unsigned long long *global_twins) {
    
    // CHUNK_SIZE + 2 do sprawdzania siostrzanych
    __shared__ char s_sieve[CHUNK_SIZE + 2];
    __shared__ unsigned int block_primes;
    __shared__ unsigned int block_twins;

    if (threadIdx.x == 0) {
        block_primes = 0;
        block_twins = 0;
    }

    // początkowa wartość, od której zaczyna się ten konkretny pakiet (np. pakiet nr 1 zaczyna się od 16384)
    long chunk_start_val = blockIdx.x * CHUNK_SIZE;
    long thread_idx = threadIdx.x;

    // pakiet wspoldzielony
    for (long i = thread_idx; i < CHUNK_SIZE + 2; i += blockDim.x) {
        long val = chunk_start_val + i;
        s_sieve[i] = (val >= 2) ? 1 : 0;
    }
    __syncthreads();

    // wykreslanie w pakiecie za pomoca malego sita
    for (long p = thread_idx; p <= max_base_val; p += blockDim.x) {
        if (p >= 2 && base_sieve[p] == 1) {
            
            long start_val = (chunk_start_val / p) * p;
            if (start_val < chunk_start_val) {
                start_val += p;
            }
            // wielokrotności zaczynamy wykreślać najwcześniej od p*p
            if (start_val < p * p) {
                start_val = p * p;
            }

            // zamiana wartości globalnej na lokalny indeks w pamięci L1 (od 0 do 16384)
            long start_local_idx = start_val - chunk_start_val;

            // wykreślanie w pamięci lokalnej
            for (long j = start_local_idx; j < CHUNK_SIZE + 2; j += p) {
                s_sieve[j] = 0;
            }
        }
    }
    __syncthreads();

    // zliczanie w pakiecie
    unsigned int local_p = 0;
    unsigned int local_t = 0;

    for (long i = thread_idx; i < CHUNK_SIZE; i += blockDim.x) {
        long val = chunk_start_val + i;
        
        if (val <= target_limit) {
            if (s_sieve[i] == 1) {
                local_p++;
                
                if (val + 2 <= target_limit) {
                    if (s_sieve[i + 2] == 1) {
                        local_t++;
                    }
                }
            }
        }
    }

    if (local_p > 0) atomicAdd(&block_primes, local_p);
    if (local_t > 0) atomicAdd(&block_twins, local_t);
    __syncthreads();

    // zrzut sum z całego bloku do pamieci globalnej vram
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
    long max_base_val = (long)sqrt((double)max_val);

    unsigned long long h_primes = 0; 
    unsigned long long h_twins = 0;

    unsigned long long *d_primes = NULL;
    unsigned long long *d_twins = NULL;
    char *d_base_sieve = NULL;

    if (cudaSuccess != cudaMalloc((void **)&d_primes, sizeof(unsigned long long))) errorexit("Error allocating primes");
    if (cudaSuccess != cudaMalloc((void **)&d_twins, sizeof(unsigned long long))) errorexit("Error allocating twins");
    
    // alokujemy bazowe sito do pierwiastka
    if (cudaSuccess != cudaMalloc((void **)&d_base_sieve, (max_base_val + 1) * sizeof(char))) errorexit("Error allocating base sieve");

    if (cudaSuccess != cudaMemcpy(d_primes, &h_primes, sizeof(unsigned long long), cudaMemcpyHostToDevice)) errorexit("Error copying");
    if (cudaSuccess != cudaMemcpy(d_twins, &h_twins, sizeof(unsigned long long), cudaMemcpyHostToDevice)) errorexit("Error copying");

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);
    
    int threadsInBlock = 1024;

    // inicjacja bazowego sita
    int blocksBaseInit = (max_base_val + 1 + threadsInBlock - 1) / threadsInBlock;
    init_base_sieve<<<blocksBaseInit, threadsInBlock>>>(d_base_sieve, max_base_val);
    cudaDeviceSynchronize();
    
    run_base_sieve<<<blocksBaseInit, threadsInBlock>>>(d_base_sieve, max_base_val);
    cudaDeviceSynchronize();

    // każdy blok dostaje chunka liczb
    int blocksSegmented = (target_limit + 1 + CHUNK_SIZE - 1) / CHUNK_SIZE; 
    segmented_sieve<<<blocksSegmented, threadsInBlock>>>(d_base_sieve, target_limit, max_base_val, d_primes, d_twins);
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