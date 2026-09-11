#include <hip/hip_runtime.h>
#include <cstdio>
typedef int __attribute__((vector_size(2 * sizeof(int)))) i32x2;
typedef int __attribute__((vector_size(4 * sizeof(int)))) i32x4;
typedef int __attribute__((vector_size(8 * sizeof(int)))) i32x8;

__global__ __launch_bounds__(64) void iu8_loop(const int8_t* a, const int8_t* b, int* out, int iters) {
  i32x4 af = {(int)threadIdx.x, 1, 2, 3};
  i32x4 bf = {(int)threadIdx.x, 1, 2, 3};
  i32x8 acc = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < iters; ++i) {
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, acc, true);
    af[0] = acc[0];
  }
  int s = 0;
  for (int i = 0; i < 8; ++i) s += acc[i];
  out[threadIdx.x] = s;
}

__global__ __launch_bounds__(64) void iu4_loop(const int8_t* a, const int8_t* b, int* out, int iters) {
  i32x2 af = {(int)threadIdx.x, 1};
  i32x2 bf = {(int)threadIdx.x, 1};
  i32x8 acc = {0,0,0,0,0,0,0,0};
  for (int i = 0; i < iters; ++i) {
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, af, true, bf, acc, true);
    af[0] = acc[0];
  }
  int s = 0;
  for (int i = 0; i < 8; ++i) s += acc[i];
  out[threadIdx.x] = s;
}

int main() {
  int8_t* a; int* out;
  hipMalloc((void**)&a, 4096);
  hipMalloc((void**)&out, 4096);
  hipMemset(a, 0, 4096);
  const int iters = 2000000;
  dim3 grid(160), block(64);
  hipEvent_t t0, t1;
  hipEventCreate(&t0); hipEventCreate(&t1);
  // warmup
  iu8_loop<<<grid, block>>>(a, a, out, 1000);
  iu4_loop<<<grid, block>>>(a, a, out, 1000);
  hipDeviceSynchronize();
  hipEventRecord(t0);
  iu8_loop<<<grid, block>>>(a, a, out, iters);
  hipEventRecord(t1);
  hipDeviceSynchronize();
  float ms8; hipEventElapsedTime(&ms8, t0, t1);
  hipEventRecord(t0);
  iu4_loop<<<grid, block>>>(a, a, out, iters);
  hipEventRecord(t1);
  hipDeviceSynchronize();
  float ms4; hipEventElapsedTime(&ms4, t0, t1);
  double ops8 = 1.0 * grid.x * block.x * iters * 16.0 * 16.0 * 16.0 / 32.0; // MACs per warp-instruction path
  double ops4 = ops8; // same instruction count path
  printf("iu8: %.1f ms, iu4: %.1f ms, ratio iu8/iu4 = %.3f (same instruction count)\n",
         ms8, ms4, ms8 / ms4);
  printf("throughput ratio iu4/iu8 per instruction: %.3f\n", ms8 / ms4);
  return 0;
}
