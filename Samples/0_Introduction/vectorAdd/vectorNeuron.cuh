#pragma once

struct ML_Neuron;
template <class Type>
struct ML_Matrix;

void Forward(ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& output);
void Backward(ML_Matrix<float>& output, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connectionDerivative, ML_Matrix<float>& inputDerivative);
void Error(ML_Matrix<float>& value, ML_Matrix<float>& expected, ML_Matrix<float>& error);

template<class Type>
__global__ void vectorApply(ML_DeviceMatrix<Type> original, const ML_DeviceMatrix<Type> deriviative, const float rate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < original.Count())
    {
        original[i] += deriviative[i] * rate;
    }
}
template<class Type>
void Apply(ML_Matrix<Type>& original, ML_Matrix<Type>& deriviative, float rate)
{
    assert(original.Dimensions() == deriviative.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ original.Dimensions() };
    vectorApply CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock) (original.DeviceArray(), deriviative.DeviceArray(), rate);

    // Debug CPU copy back
    original.HostArray();
}