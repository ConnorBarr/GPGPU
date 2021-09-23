#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include "device_launch_parameters.h"
__host__ __device__ unsigned int get_num_threads()
{
#if !defined(__CUDA_ARCH__)
	std::printf("HOST DEVICE FUNCTION WHEN RAN ON CPU/HOST RETURNS = ERROR:CODE EXECUTED ON CPU/HOST \n");
	return 1;
#else
	
	unsigned int nthreads = gridDim.x * blockDim.x;
	if (threadIdx.x == 3 && blockIdx.x == 2)
	{
		std::printf("HOST DEVICE FUNCTION WHEN RAN ON DEVICE RETURNS = ");
		std::printf("Number of Threads: %i \n", nthreads);
	}
	else {}
	return nthreads;

#endif;

}
__global__ void simple_cuda()
{
	std::printf("Block: %i Thread: %i \n", blockIdx.x, threadIdx.x);
	if(blockIdx.x == 2 && threadIdx.x == 3)
	{
		std::printf("Blocks Per Grid: %i Threads per Block: %i \n", gridDim.x, blockDim.x);
	}
	

	get_num_threads(); //Return Proper due to being ran on GPU
}
int main(int argc, char* argv[])
{
	std::string t = cudaGetErrorString(cudaSetDevice(0));
	if(t == "no error")
	{
		std::cout << "Device ID valid \n";
	}
	else
	{
		std::cout << "Device ID Invalid \n";
	}

	int i;
	cudaDeviceProp CurrentGPU;


	cudaGetDeviceCount(&i);
	std::cout << "Number of Devices: "<< i << " \n";
	cudaGetDeviceProperties(&CurrentGPU, 0);
	std::cout <<  CurrentGPU.name << "\n";
	std::cout << "Compute Compatability: " << CurrentGPU.major << "." << CurrentGPU.minor << "\n";

	simple_cuda<<<3,4>>>(); //12threads

	cudaDeviceSynchronize(); //Prevents Running of other Functions in Pipeline until all before it are complete
	get_num_threads(); //Will return 1 because ran on CPU
	
	


 	return 0;
}