#include <iostream>
#include <vector>
#include <cassert>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include<cstdio>

template <unsigned BLOCK_SIZE>
__global__ void sync(float *f)
{
  __shared__ float tmp[BLOCK_SIZE];
  unsigned local_id = blockIdx.x * blockDim.x + threadIdx.x;
  tmp[local_id] = static_cast<float>(local_id);


  // ?? add something here ??
  __syncthreads();

  f[local_id] = tmp[BLOCK_SIZE-1-local_id];
}

int main(int argc, char* argv[])
{
    const unsigned block_size{ 128 };
    float* d_p{ nullptr };
    std::vector<float> v(block_size, 0); // set vector size and zero it
    const unsigned nbytes = block_size * sizeof(unsigned);

    // ?? add something here ??
    cudaMalloc((void**)&d_p, nbytes);
    cudaMemcpy(d_p, v.data(), nbytes, cudaMemcpyHostToDevice);

    sync<block_size> << <1, block_size >> > (d_p);

    cudaMemcpy(v.data(), d_p, nbytes, cudaMemcpyDeviceToHost);

    // there is something wrong with the code on the next 3 lines
    bool b = false;
    for (unsigned i = 0; i < block_size; ++i)
        if (i == v[block_size - 1 - i])
        {
            b = true;
        }
        else
        {
            b = false;
            break;
        }

  assert(b && "The elements of v should be 127,126,125,124...");

  

  cudaFree(d_p);

  return 0;
}
