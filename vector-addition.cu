#include <cuda_runtime.h>
#include <cmath> // for ceil if needed

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    int n = 1 << 10; // 1024 elements
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    for (int i = 0; i < n; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    vecAdd(A, B, C, n);

    // Print first 5 results
    for (int i = 0; i < 5; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}