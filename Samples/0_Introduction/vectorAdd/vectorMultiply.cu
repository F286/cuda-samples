#pragma once
#include "ML_Helpers.h"
#include "ML_Array.h"

__global__ void vectorMultiply(const ML_DeviceMatrix<float> input, const ML_DeviceMatrix<float> connection, ML_DeviceMatrix<float> output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < output.Count())
    {
        const float* connectionStart = &connection[i * input.dimensions.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < input.dimensions.x; elementIndex++)
        {
            total += input[elementIndex] * connectionStart[elementIndex];
        }
        output[i] = total;
    }
}
void Multiply(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output)
{
    ML_Helpers::VerifyForwardConnection(input.Dimensions(), connection.Dimensions(), output.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ output.Dimensions() };
    vectorMultiply CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), connection.DeviceArray(), output.DeviceArray());
}

