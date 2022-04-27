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
}

//__global__ void vectorBackward(const ML_DeviceMatrix<float> source, const ML_DeviceMatrix<ML_Neuron> connection, ML_DeviceMatrix<ML_Neuron> derivative) {
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (i < derivative.Count())
//    {
//        const int sourceIndex = i % connection.Dimensions().y;
//
//        ML_Neuron set;
//
//        set.bias = source[sourceIndex] - connection[i].bias;
//        set.weight = set.bias / connection[i].weight;
//
//        derivative[i] = set;
//    }
//}
//void Backward(ML_Matrix<float>& source, ML_Matrix<ML_Neuron>& connection, ML_Matrix<ML_Neuron>& derivative)
//{
//    ML_Helpers::VerifyBackwardConnection(source.Dimensions(), connection.Dimensions(), derivative.Dimensions());
//    ML_CheckCudaError checkError;
//    ML_KernelSize size{ derivative.Dimensions() };
//    vectorBackward CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(source.DeviceArray(), connection.DeviceArray(), derivative.DeviceArray());
//}

__global__ void vectorBackward(const ML_DeviceMatrix<float> output, const ML_DeviceMatrix<ML_Neuron> connection, ML_DeviceMatrix<ML_Neuron> connectionDerivative, ML_DeviceMatrix<float> inputDerivative) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < connection.Count())
    {
        const int sourceIndex = i / connection.Dimensions().x;

        // Partial derivative, using chain rule
        ML_Neuron partial;

        partial.bias = output[sourceIndex] - connection[i].bias;
        partial.weight = partial.bias / connection[i].weight;

        connectionDerivative[i] = partial;

        const int derivativeIndex = i % connection.Dimensions().x;
        atomicAdd(&inputDerivative[derivativeIndex], partial.weight);
    }
}
void Backward(ML_Matrix<float>& output, ML_Matrix<ML_Neuron>& connection, ML_Matrix<ML_Neuron>& connectionDerivative, ML_Matrix<float>& inputDerivative)
{
    assert(output.Dimensions().x == connection.Dimensions().y);
    assert(connection.Dimensions().x == inputDerivative.Dimensions().x);
    assert(connection.Dimensions() == connectionDerivative.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ inputDerivative.Dimensions() };
    vectorBackward CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(output.DeviceArray(), connection.DeviceArray(), connectionDerivative.DeviceArray(), inputDerivative.DeviceArray());
}

__global__ void vectorError(const ML_DeviceMatrix<float> input, const ML_DeviceMatrix<float> expected, ML_DeviceMatrix<float> error) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < error.Count())
    {
        //const int sourceIndex = i / connection.Dimensions().x;

        //// Partial derivative, using chain rule
        //ML_Neuron partial;

        //partial.bias = output[sourceIndex] - connection[i].bias;
        //partial.weight = partial.bias / connection[i].weight;

        //connectionDerivative[i] = partial;

        //const int derivativeIndex = i % connection.Dimensions().x;
        //atomicAdd(&inputDerivative[derivativeIndex], partial.weight);
    }
}
void Error(ML_Matrix<float>& input, ML_Matrix<float>& expected, ML_Matrix<float>& error)
{
    assert(input.Dimensions() == expected.Dimensions());
    assert(input.Dimensions() == error.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ error.Dimensions() };
    vectorError CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), expected.DeviceArray(), error.DeviceArray());
}