// refQ_img.c -- independent exact computation of the image and multiplicity
// histogram of f(u) = sigma0(u) - u mod 2^32 over the full domain.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static inline uint32_t rotr(uint32_t x, int n){ return (x >> n) | (x << (32 - n)); }
static inline uint32_t s0(uint32_t x){ return rotr(x,7) ^ rotr(x,18) ^ (x >> 3); }

int main(int argc, char **argv){
    int mode = (argc > 1) ? atoi(argv[1]) : 0;   // 0: sigma0-u  1: sigma0+u  2: sigma1-u  3: sigma0 xor u
    size_t N = (size_t)1 << 32;
    uint8_t *cnt = mmap(NULL, N, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (cnt == MAP_FAILED){ perror("mmap"); return 1; }
    madvise(cnt, N, MADV_HUGEPAGE);
    clock_t t0 = clock();
    uint64_t u = 0;
    do {
        uint32_t x = (uint32_t)u, v;
        switch(mode){
          case 1: v = s0(x) + x; break;
          case 2: v = (rotr(x,17)^rotr(x,19)^(x>>10)) - x; break;
          case 3: v = s0(x) ^ x; break;
          default: v = s0(x) - x;
        }
        if (cnt[v] < 255) cnt[v]++;
        u++;
    } while (u < N);
    fprintf(stderr, "scan done %.0fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    uint64_t hist[256]; memset(hist, 0, sizeof hist);
    for (uint64_t i = 0; i < N; i++) hist[cnt[i]]++;
    uint64_t ge1=0, ge2=0, ge3=0, ge4=0, tot=0;
    for (int k = 0; k < 256; k++){
        if (k>=1) ge1 += hist[k];
        if (k>=2) ge2 += hist[k];
        if (k>=3) ge3 += hist[k];
        if (k>=4) ge4 += hist[k];
        tot += (uint64_t)k * hist[k];
    }
    printf("mode=%d\n", mode);
    printf("total_preimages_counted = %llu (should be 4294967296)\n", (unsigned long long)tot);
    printf("image(>=1) = %llu  frac=%.9f\n", (unsigned long long)ge1, ge1/4294967296.0);
    printf(">=2        = %llu  frac=%.9f\n", (unsigned long long)ge2, ge2/4294967296.0);
    printf(">=3        = %llu  frac=%.9f\n", (unsigned long long)ge3, ge3/4294967296.0);
    printf(">=4        = %llu  frac=%.9f\n", (unsigned long long)ge4, ge4/4294967296.0);
    for (int k = 0; k <= 16; k++)
        printf("  exactly %2d : %llu  frac=%.9f\n", k, (unsigned long long)hist[k], hist[k]/4294967296.0);
    // does the max root of some value equal 0xFFFFFFFF (sentinel clash)?
    uint32_t vlast; { uint32_t x=0xFFFFFFFFu; vlast = s0(x)-x; }
    printf("f(0xFFFFFFFF) = %u  multiplicity there = %d\n", vlast, cnt[vlast]);
    return 0;
}
