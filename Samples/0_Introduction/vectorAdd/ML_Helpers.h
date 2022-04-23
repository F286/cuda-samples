#pragma once
#include "ML_Array.h"
#include <crt/common_functions.h>

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

namespace ML_Helpers
{
    void VerifyForwardConnection(Int2 input, Int2 connection, Int2 output)
    {
        assert(connection.x == input.x);
        assert(connection.y == output.x);
    }
}

struct ML_KernelSize
{
    ML_KernelSize(const Int2 dimensions)
        : threadsPerBlock(256)
        , blocksPerGrid((dimensions.Size() + threadsPerBlock - 1) / threadsPerBlock)

    {
    }

    int threadsPerBlock;
    int blocksPerGrid;
};

