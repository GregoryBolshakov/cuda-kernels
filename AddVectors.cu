#include <iostream>

__global__ void addVectors(float* a, float* b, float* c, int n) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
    if (i + 1< n) {
        c[i + 1] = a[i + 1] + b[i + 1];
    }
}

int main() {
    float vecA_h[1000];
    float vecB_h[1000];
    float vecC_h[1000];
    for (int i = 0; i < 1000; ++i) {
        vecA_h[i] = rand() % 1001;
        vecB_h[i] = rand() % 1001;
    }
    int n = 1000;
    float* vecA_d;
    float* vecB_d;
    float* vecC_d;
    int size = n * sizeof(float);
    
    cudaMalloc(&vecA_d, size);
    cudaMalloc(&vecB_d, size);
    cudaMalloc(&vecC_d, size);

    cudaMemcpy(vecA_d, vecA_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(vecB_d, vecB_h, size, cudaMemcpyHostToDevice);

    // Call kernel function
    addVectors<<<(n + 511) / (512), 256>>>(vecA_d, vecB_d, vecC_d, n);

    cudaMemcpy(vecC_h, vecC_d, size, cudaMemcpyDeviceToHost);

    cudaFree(vecA_d);
    cudaFree(vecB_d);
    cudaFree(vecC_d);

    for (int i = 0; i < 1000; ++i) {
        std::cout << vecC_h[i] << " ";
    }

    return 0;
}
