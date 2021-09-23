#include <iostream>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
  c[index] = a[index] + b[index];
}

int main(int argc, char *argv[])
{
  const int N = 2*3*4*2*3*4; // 576;
  int *a,     *b,   *c;   // host   copies of a, b, c
  int *d_a, *d_b, *d_c;   // device copies of a, b, c
  int nbytes = N * sizeof(int);

  // Alloc space for device copies of a, b, c
  cudaMalloc((void **)&d_a, nbytes);
  cudaMalloc((void **)&d_b, nbytes);
  cudaMalloc((void **)&d_c, nbytes);

  // Alloc space for host copies of a, b, c and setup input values
  a = new int[N];
  b = new int[N];
  c = new int[N];

  // Initialise a and b to a simple arithmetic sequence
  for (int i = 0; i < N; i++) { a[i] = i; b[i] = i; }

  // Copy inputs to device
  cudaMemcpy(d_a, a, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, nbytes, cudaMemcpyHostToDevice);

  // Launch add() kernel on GPU with N blocks
  add<<<24,24>>>(d_a, d_b, d_c);

  // No need for a cudaDeviceSynchronize here: CUDA operations issued
  // to the same stream (here the default one) *always* serialize.

  // Copy result back to host
  cudaMemcpy(c, d_c, nbytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) { std::cout << c[i] << ','; }
  std::cout << '\n';

  // Cleanup
  delete [] a; delete [] b; delete [] c;
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
