#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

struct ML_KernelSize
{
    ML_KernelSize(const Int2 numElements)
        : threadsPerBlock(256)
        , blocksPerGrid((numElements.Size() + threadsPerBlock - 1) / threadsPerBlock)

    {
    }

    int threadsPerBlock;
    int blocksPerGrid;
};

