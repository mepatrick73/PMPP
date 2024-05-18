#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <time.h>

void vec_add(float *A_h, float *B_h, float *C_h, int n) {
  for (int i = 0; i < n; ++i) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vec_add_gpu(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_h, size);
  cudaMalloc((void **)&B_h, size);
  cudaMalloc((void **)&C_h, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // Kernel invocation
  vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

bool vec_compare(float *A_h, float *B_h, int n, float epsilon) {
  for (int i = 0; i < n; ++i) {
    if (fabs(A_h[i] - B_h[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

void gen_random_vec(float *random_vec, int n) {
  srand((unsigned int)time(NULL));
  float MAX_RANDOM_VALUE = 1000.0;
  for (int i = 0; i < n; ++i) {
    random_vec[i] = (float)rand() / (float)(RAND_MAX / MAX_RANDOM_VALUE);
  }
}

int main() {
  int n = 10000;
  float A[n];
  float B[n];
  float C_cpu[n];
  float C_gpu[n];

  gen_random_vec(A, n);
  gen_random_vec(B, n);

  vec_add(A, B, C_cpu, n);
  vec_add_gpu(A, B, C_gpu, n);

  float precision = 0.000001;
  bool are_gpu_cpu_vec_equals = vec_compare(C_cpu, C_gpu, n, precision);

  if (are_gpu_cpu_vec_equals) {
    printf("cpu function and gpu kernels do the same thing !!!!!");
  } else {
    printf("cpu function and gpu kernels don't do the same thing :(");
  }
}
