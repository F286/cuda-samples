#pragma once
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <memory>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#include "ML_CheckCudaError.h"
#include "ML_Array.h"

namespace ML_Helpers
{
    void VerifyForwardConnection(Int2 input, Int2 connection, Int2 output);
    void VerifyBackwardConnection(Int2 source, Int2 connection, Int2 derivative);
}

struct ML_KernelSize
{
    ML_KernelSize(const Int2 dimensions)
        : threadsPerBlock(256)
        , blocksPerGrid((dimensions.Count() + threadsPerBlock - 1) / threadsPerBlock)

    {
    }

    int threadsPerBlock;
    int blocksPerGrid;
};

