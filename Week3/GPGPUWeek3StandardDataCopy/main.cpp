#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include "device_launch_parameters.h"
#include <vector>
__host__ __device__ unsigned int get_num_threads()
{
#if !defined(__CUDA_ARCH__)
	std::printf("HOST DEVICE FUNCTION WHEN RAN ON CPU/HOST RETURNS = ERROR:CODE EXECUTED ON CPU/HOST \n");
	return 1;
#else
	
	unsigned int nthreads = (blockDim.x * blockDim.y) * (gridDim.y * gridDim.x);
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
	{
		std::printf("HOST DEVICE FUNCTION WHEN RAN ON DEVICE RETURNS = ");
		std::printf("Number of Threads: %i \n", nthreads);
	}
	else {}
	return nthreads;

#endif;

}

__device__ int getGlobalIdx_2D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
__global__ void simple_cuda(unsigned int* d_p)
{
	unsigned int x = threadIdx.x + (blockIdx.x * blockDim.x);
	std::printf("BlockX: %i BlockY: %i ThreadX: %i ThreadY: %i Unique Number: %i \n", blockIdx.x, blockIdx.y, threadIdx.x,threadIdx.y, getGlobalIdx_2D_2D());
	
	if(blockIdx.x == 2 && threadIdx.x == 3)
	{
		std::printf("Blocks Per Grid: %i Threads per Block: %i \n", gridDim.x, blockDim.x);
	}
	
	d_p[getGlobalIdx_2D_2D()] = getGlobalIdx_2D_2D();
	
	
	get_num_threads(); //Return Proper due to being ran on GPU
}
int main(int argc, char* argv[])
{
	std::string t = cudaGetErrorString(cudaSetDevice(0));
	if (t == "no error")
	{
		std::cout << "Device ID valid \n";
	}
	else
	{
		std::cout << "Device ID Invalid \n";
	}

	int i;
	std::vector<unsigned int>h_p(120);
	unsigned int* d_p = new unsigned int[120];
	d_p[0] = 100;
	int size = 960;

	cudaMalloc((void**)&d_p, size);
	cudaMemcpy(d_p, h_p.data(), size, cudaMemcpyHostToDevice);
	std::cout << sizeof(h_p) << h_p[119]<< std::endl;





	cudaDeviceProp CurrentGPU;


	cudaGetDeviceCount(&i);
	std::cout << "Number of Devices: "<< i << " \n";
	cudaGetDeviceProperties(&CurrentGPU, 0);
	std::cout <<  CurrentGPU.name << "\n";
	std::cout << "Compute Compatability: " << CurrentGPU.major << "." << CurrentGPU.minor << "\n";

	simple_cuda<<<dim3(2,3),dim3(4,5)>>>(d_p); //120threads
	cudaMemcpy(h_p.data(), d_p, size, cudaMemcpyDeviceToHost);
	
	for(int v = 0;v<120;v++)
	{
		std::cout << h_p[v] << std::endl;
	}

	cudaDeviceSynchronize(); //Prevents Running of other Functions in Pipeline until all before it are complete
	get_num_threads(); //Will return 1 because ran on CPU



	cudaFree(d_p);
	
	


 	return 0;
}