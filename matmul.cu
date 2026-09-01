// matmul.cu
// HW1 Part 3 — Matrix multiplication in CUDA C, benchmarked against a CPU baseline.
// SID4=6359 SEED=6359 HP_ID=5 (Schedule-long; unused by this CUDA task, reported per Sec 0.1)
//
// Build (Colab GPU runtime):
//   nvcc -O3 -Xcompiler -fopenmp matmul.cu -o matmul
// Run:
//   ./matmul 256
//   ./matmul 1024
//   ./matmul 4096
//
// Blocks/threads design (see kernel comment below): each CUDA thread computes exactly
// one output element C[row][col]. Threads are grouped into BLOCK_SIZE x BLOCK_SIZE
// (16x16 = 256 threads) thread blocks; a block therefore computes a 16x16 tile of C.
// The grid is sized as ceil(N/16) x ceil(N/16) blocks so the whole N x N output is
// covered, including when N is not a multiple of 16 (bounds-checked in the kernel).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// ---------------------------------------------------------------------------
// CUDA kernel: naive tiled-by-thread-block matrix multiplication, C = A * B
// A: N x N, B: N x N, C: N x N (square matrices, row-major, float32)
//
// Grid/block mapping:
//   - blockDim = (BLOCK_SIZE, BLOCK_SIZE) -> 256 threads per block
//   - gridDim  = (ceil(N/BLOCK_SIZE), ceil(N/BLOCK_SIZE)) blocks
//   - each block owns a BLOCK_SIZE x BLOCK_SIZE tile of the output matrix C
//   - within a block, thread (tx, ty) computes exactly one element:
//         col = blockIdx.x * BLOCK_SIZE + threadIdx.x
//         row = blockIdx.y * BLOCK_SIZE + threadIdx.y
//     that thread walks the full K-dimension (K = N here) doing a dot product
//     of A's row `row` against B's column `col`, then writes C[row][col].
// ---------------------------------------------------------------------------
__global__ void matmulKernel(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

static void cpuMatmul(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }
}

static void fillRandom(float* M, size_t count, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < count; ++i) {
        M[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

static double maxAbsDiff(const float* a, const float* b, size_t count) {
    double m = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),   \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

int main(int argc, char** argv) {
    const unsigned SEED = 6359; // SEED = SID4, used to seed matrix generation
    int N = 256;
    if (argc > 1) N = atoi(argv[1]);
    size_t bytes = (size_t)N * N * sizeof(float);
    printf("N = %d (%.2f MB per matrix)\n", N, bytes / (1024.0 * 1024.0));

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_cpu = (float*)malloc(bytes);
    float *hC_gpu = (float*)malloc(bytes);
    fillRandom(hA, (size_t)N * N, SEED);
    fillRandom(hB, (size_t)N * N, SEED + 1);

    // ---- CPU baseline timing ----
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuMatmul(hA, hB, hC_cpu, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

    // ---- GPU: allocate device memory ----
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t h2dStart, h2dStop, kStart, kStop, d2hStart, d2hStop;
    cudaEventCreate(&h2dStart); cudaEventCreate(&h2dStop);
    cudaEventCreate(&kStart);   cudaEventCreate(&kStop);
    cudaEventCreate(&d2hStart); cudaEventCreate(&d2hStop);

    // Host -> Device transfer
    cudaEventRecord(h2dStart);
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(h2dStop);
    cudaEventSynchronize(h2dStop);
    float h2dMs = 0;
    cudaEventElapsedTime(&h2dMs, h2dStart, h2dStop);

    // Kernel launch: block/grid dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("threadsPerBlock = (%d, %d) = %d threads/block\n",
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.x * threadsPerBlock.y);
    printf("numBlocks       = (%d, %d) = %d blocks\n",
           numBlocks.x, numBlocks.y, numBlocks.x * numBlocks.y);

    // warm-up launch (not timed) to exclude first-launch overhead from the measurement
    matmulKernel<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(kStart);
    matmulKernel<<<numBlocks, threadsPerBlock>>>(dA, dB, dC, N);
    cudaEventRecord(kStop);
    cudaEventSynchronize(kStop);
    CUDA_CHECK(cudaGetLastError());
    float kernelMs = 0;
    cudaEventElapsedTime(&kernelMs, kStart, kStop);

    // Device -> Host transfer
    cudaEventRecord(d2hStart);
    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(d2hStop);
    cudaEventSynchronize(d2hStop);
    float d2hMs = 0;
    cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop);

    double transferMs = h2dMs + d2hMs;
    double gpuEndToEndMs = kernelMs + transferMs;
    double speedup = cpuMs / gpuEndToEndMs;

    double diff = maxAbsDiff(hC_cpu, hC_gpu, (size_t)N * N);

    printf("\n---- RESULTS (N=%d) ----\n", N);
    printf("CPU time (ms):           %.4f\n", cpuMs);
    printf("GPU kernel time (ms):    %.4f\n", kernelMs);
    printf("H2D+D2H transfer (ms):   %.4f\n", transferMs);
    printf("GPU end-to-end (ms):     %.4f\n", gpuEndToEndMs);
    printf("Speedup (CPU/GPU e2e):   %.4fx\n", speedup);
    printf("Max abs diff CPU vs GPU: %e\n", diff);
    printf("CSV,%d,%.4f,%.4f,%.4f,%.4f\n", N, cpuMs, kernelMs, transferMs, speedup);

    cudaEventDestroy(h2dStart); cudaEventDestroy(h2dStop);
    cudaEventDestroy(kStart);   cudaEventDestroy(kStop);
    cudaEventDestroy(d2hStart); cudaEventDestroy(d2hStop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_cpu); free(hC_gpu);
    return 0;
}
