#include <cmath>
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <time.h>

const int CHANNELS = 3;

// rgb_h is assumed to have 3 channels for r g  and b
void cpu_greyscale(unsigned char *rgb_h, unsigned char *grey_h, int width,
                   int height) {
  for (int grey_offset = 0; grey_offset < width * height; ++grey_offset) {
    int rgb_offset = grey_offset * CHANNELS;
    unsigned char r = rgb_h[rgb_offset];
    unsigned char g = rgb_h[rgb_offset + 1];
    unsigned char b = rgb_h[rgb_offset + 2];
    grey_h[grey_offset] = 0.21 * r + 0.72 * g + 0.07 * b;
  }
}

__global__ void greyscale_kernel(unsigned char *rgb, unsigned char *grey,
                                 int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < height && col < width) {
    int grey_offset = row * width + col;
    int rgb_offset = grey_offset * CHANNELS;
    unsigned char r = rgb[rgb_offset];
    unsigned char g = rgb[rgb_offset + 1];
    unsigned char b = rgb[rgb_offset + 2];
    grey[grey_offset] = 0.21 * r + 0.72 * g + 0.07 * b;
  }
}

void gpu_greyscale(unsigned char *rgb_h, unsigned char *grey_h, int width,
                   int height) {
  int size_grey = width * height * sizeof(unsigned char);
  int size_rgb = size_grey * CHANNELS;
  unsigned char *rgb_d, *grey_d;

  cudaMalloc((void **)&grey_d, size_grey);
  cudaMalloc((void **)&rgb_d, size_rgb);

  cudaMemcpy(rgb_d, rgb_h, size_rgb, cudaMemcpyHostToDevice);

  // setup grid/block dimensions
  dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
  dim3 dimBlock(16, 16, 1);

  greyscale_kernel<<<dimGrid, dimBlock>>>(rgb_d, grey_d, width, height);

  cudaMemcpy(grey_h, grey_d, size_grey, cudaMemcpyDeviceToHost);

  cudaFree(rgb_d);
  cudaFree(grey_d);
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
  float MAX_RANDOM_VALUE = 1000.0;
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
  int width = 100;
  int height = 100;
  unsigned char rgb[width * height * CHANNELS];
  unsigned char grey_cpu[width * height];
  unsigned char grey_gpu[width * height];

  gen_random_uc_vec(rgb, width * height * CHANNELS);

  cpu_greyscale(rgb, grey_cpu, width, height);
  gpu_greyscale(rgb, grey_gpu, width, height);

  float precision = 0.000001;
  bool are_gpu_cpu_vec_equals =
      vec_uc_compare(grey_gpu, grey_cpu, width * height);

  if (are_gpu_cpu_vec_equals) {
    printf("cpu function and gpu kernels do the same thing !!!!!\n");
  } else {
    printf("cpu function and gpu kernels don't do the same thing :(\n");
  }
}
