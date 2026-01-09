#include <cuda_runtime.h>

__global__ void add_kernel(float *a, float *b, float *c)
{
    *c = *a + *b;
}

extern "C" __declspec(dllexport)
void add_cuda(float *a, float *b, float *result)
{
    cudaSetDevice(0);
    float* d_a = nullptr;
    float* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    
    cudaMemcpy(d_a,a,sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_c = nullptr;
    cudaMalloc(&d_c, sizeof(float));

    add_kernel<<<1, 1>>>(d_a, d_b, d_c);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

}
