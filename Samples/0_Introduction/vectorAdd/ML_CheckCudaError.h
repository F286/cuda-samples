#pragma once
#include <cuda_runtime.h>
#include <helper_cuda.h>

struct ML_CheckCudaError
{
    ~ML_CheckCudaError()
    {
        cudaError_t err = cudaSuccess;
        err = cudaGetLastError();

        if (err != cudaSuccess) {

            fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
            __debugbreak();
            exit(EXIT_FAILURE);
        }
    }
};