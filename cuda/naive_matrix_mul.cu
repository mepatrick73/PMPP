#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// A * B = C where A is of dim m x n and B is of dim n x p
// Matrix are row major
void naive_cpu_mat_mul(float *A, float *B, float *C, int m, int n, int p) {
  for (int out_row = 0; out_row < m; ++out_row) {
    for (int out_col = 0; out_col < p; ++out_col) {
      float sum = 0.0f;
      for (int index = 0; index < n; index++) {
        sum += A[out_row * n + index] * B[index * n + out_col];
      }
      C[out_row * p + out_col] = sum;
    }
  }
}

__global__ void naive_gpu_mat_mul_kernel(float *A, float *B, float *C, int m,
                                         int n, int p) {
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_row < m && out_col < p) {
    float sum = 0.0f;
    for (int index = 0; index < n; index++) {
      sum += A[out_row * n + index] * B[index * n + out_col];
    }
    C[out_row * p + out_col] = sum;
  }
}

void naive_gpu_mat_mul(float *A_h, float *B_h, float *C_h, int m, int n,
                       int p) {
  int size_A = m * n * sizeof(float);
  int size_B = n * p * sizeof(float);
  int size_C = m * p * sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, size_A);
  cudaMalloc((void **)&B_d, size_B);
  cudaMalloc((void **)&C_d, size_C);

  cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

  // setup grid/block dimensions
  dim3 dimGrid(ceil(p / 16.0), ceil(m / 16.0), 1);
  dim3 dimBlock(16, 16, 1);

  naive_gpu_mat_mul_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, m, n, p);

  cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

bool vec_compare(float *A_h, float *B_h, int n, float epsilon) {
  for (int i = 0; i < n; ++i) {
    if (fabs(A_h[i] - B_h[i]) > epsilon) {
      printf("A : %2.6f and B : %2.6f at position %d", A_h[i], B_h[i], i);
      return false;
    }
  }
  return true;
}

bool vec_uc_compare(unsigned char *A, unsigned char *B, int n) {
  for (int i = 0; i < n; ++i) {
    if (abs(A[i] - B[i]) > 1) {
      printf("A : %d and B : %d at position %d\n", A[i], B[i], i);
      return false;
    }
  }
  return true;
}

void gen_random_vec(float *random_vec, int n) {
  srand((unsigned int)time(NULL));
  float MAX_RANDOM_VALUE = 100.0;
  for (int i = 0; i < n; ++i) {
    random_vec[i] = (float)rand() / (float)(RAND_MAX / MAX_RANDOM_VALUE);
  }
}

void gen_random_uc_vec(unsigned char *random_vec, int n) {
  srand((unsigned int)time(NULL));
  for (int i = 0; i < n; ++i) {
    random_vec[i] = (unsigned char)(rand() % 256);
  }
}

int main() {
  int m = 2000;
  int n = 1000;
  int p = 1500;
  float *A_h, *B_h, *C_cpu_h, *C_gpu_h;
  A_h = (float *)malloc(m * n * sizeof(float));
  B_h = (float *)malloc(n * p * sizeof(float));
  C_cpu_h = (float *)malloc(m * p * sizeof(float));
  C_gpu_h = (float *)malloc(m * p * sizeof(float));

  gen_random_vec(A_h, m * n);
  gen_random_vec(B_h, n * p);

  naive_cpu_mat_mul(A_h, B_h, C_cpu_h, m, n, p);
  naive_gpu_mat_mul(A_h, B_h, C_gpu_h, m, n, p);

  float precision = 100;
  bool are_gpu_cpu_vec_equals = vec_compare(C_gpu_h, C_cpu_h, m * p, precision);

  if (are_gpu_cpu_vec_equals) {
    printf("cpu function and gpu kernels do the same thing !!!!!\n");
  } else {
    printf("cpu function and gpu kernels don't do the same thing :(\n");
  }

  free(A_h);
  free(B_h);
  free(C_cpu_h);
  free(C_gpu_h);
}
