#pragma once
#include "ML_Helpers.h"
#include "ML_Array.h"
#include "ML_Neuron.h"
#include <sm_60_atomic_functions.h>

__global__ void vectorForward(const ML_DeviceMatrix<float> input, const ML_DeviceMatrix<ML_Neuron> connection, ML_DeviceMatrix<float> output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < output.Count())
    {
        const ML_Neuron* connectionStart = &connection[i * input.dimensions.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < input.dimensions.x; elementIndex++)
        {
            const ML_Neuron& element = *(connectionStart + elementIndex);
            total += input[elementIndex] * element.weight + element.bias;
        }
        // ReLU
        output[i] = max(0.0f, total);
    }
}
void Forward(ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& output)
{
    ML_Helpers::VerifyForwardConnection(input.Dimensions(), connection.Dimensions(), output.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ output.Dimensions() };
    vectorForward CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), connection.DeviceArray(), output.DeviceArray());

    // Debug CPU copy back
    output.HostArray();
}

__global__ void vectorBackward(const ML_DeviceMatrix<float> output, const ML_DeviceMatrix<ML_Neuron> connection, const ML_DeviceMatrix<float> input, ML_DeviceMatrix<ML_Neuron> connectionDerivative, ML_DeviceMatrix<float> inputDerivative) {
    int connectionIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (connectionIndex < connection.Count())
    {
        const int outputIndex = connectionIndex % output.Dimensions().x;
        const int inputIndex = connectionIndex % input.Dimensions().x;

        // Partial derivative, using chain rule
        ML_Neuron partial;

        partial.bias = output[outputIndex];
        // Weights need to be scaled by input values and summed to get total weight derivative
        float inputTimesWeightSummed = 0.0f;
        for (int inputIndex = 0; inputIndex < input.Dimensions().x; inputIndex++)
        {
            inputTimesWeightSummed += input[inputIndex] / connection[connectionIndex].weight;
        }
        partial.weight = partial.bias / inputTimesWeightSummed;
        //partial.weight = partial.bias / connection[i].weight;

        connectionDerivative[connectionIndex] = partial;
        
        atomicAdd(&inputDerivative[inputIndex], partial.weight);
    }
}
void Backward(ML_Matrix<float>& output, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connectionDerivative, ML_Matrix<float>& inputDerivative)
{
    assert(output.Dimensions().x == connection.Dimensions().y);
    assert(connection.Dimensions().x == inputDerivative.Dimensions().x);
    assert(connection.Dimensions() == connectionDerivative.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ connectionDerivative.Dimensions() };
    vectorBackward CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(output.DeviceArray(), connection.DeviceArray(), input.DeviceArray(), connectionDerivative.DeviceArray(), inputDerivative.DeviceArray());

    // Debug CPU copy back
    connectionDerivative.HostArray();
    inputDerivative.HostArray();
}

__global__ void vectorError(const ML_DeviceMatrix<float> value, const ML_DeviceMatrix<float> expected, ML_DeviceMatrix<float> error) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < error.Count())
    {
        error[i] = expected[i] - value[i];
    }
}
void Error(ML_Matrix<float>& value, ML_Matrix<float>& expected, ML_Matrix<float>& error)
{
    assert(value.Dimensions() == expected.Dimensions());
    assert(value.Dimensions() == error.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ error.Dimensions() };
    vectorError CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(value.DeviceArray(), expected.DeviceArray(), error.DeviceArray());

    // Debug CPU copy back
    error.HostArray();
}