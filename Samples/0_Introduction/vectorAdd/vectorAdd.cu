#include <stdio.h>
#include <vector>
#include <assert.h>
#include <memory>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ML_Array.h"
#include "ML_CheckCudaError.h"
#include "ML_DenseConnection.h"
#include "ML_Helpers.h"

struct ML_Neuron
{
    float weight;
    float bias;
    // Also uses ReLU
};

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

__global__ void vectorDivide(const ML_DeviceMatrix<float> input, const ML_DeviceMatrix<float> connection, ML_DeviceMatrix<float> output) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < output.Count())
    {
        const float* connectionStart = &connection[i * input.dimensions.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < input.dimensions.x; elementIndex++)
        {
            float b = connectionStart[elementIndex];
            if (b != 0)
            {
                total += input[elementIndex] / b;
            }
        }
        output[i] = total;
    }
}
void Divide(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output)
{
    ML_Helpers::VerifyForwardConnection(input.Dimensions(), connection.Dimensions(), output.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ output.Dimensions() };
    vectorDivide CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), connection.DeviceArray(), output.DeviceArray());
}

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

__global__ void vectorBackward(const ML_DeviceMatrix<float> source, const ML_DeviceMatrix<ML_Neuron> connection, ML_DeviceMatrix<ML_Neuron> derivative) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < derivative.Count())
    {
        const int sourceIndex = i % connection.Dimensions().y;

        ML_Neuron set;

        set.bias = source[sourceIndex] - connection[i].bias;
        set.weight = set.bias / connection[i].weight;

        derivative[i] = set;
    } 
}
void Backward(ML_Matrix<float>& source, ML_Matrix<ML_Neuron>& connection, ML_Matrix<ML_Neuron>& derivative)
{
    ML_Helpers::VerifyBackwardConnection(source.Dimensions(), connection.Dimensions(), derivative.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ derivative.Dimensions() };
    vectorBackward CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(source.DeviceArray(), connection.DeviceArray(), derivative.DeviceArray());
}


void RunNetwork()
{
    ML_Matrix<float> layer1{ Int2{ 2, 1 }, {10, 100} };

    ML_Matrix<float> layer2{ Int2{ 1, 1 } };

    ML_Matrix<ML_Neuron> connection1to2{ Int2{ 2, 1 }, { {0.1, 0}, {0.1, 2} } };

    // Run forward
    Forward(layer1, connection1to2, layer2);

    assert(layer2[0] == (1 + 12));

    // Run back propagation
    ML_Matrix<float> errorLayer2{ Int2{ 1, 1 }, { 1 } };
    ML_Matrix<ML_Neuron> derivativeConnection1to2{ Int2{ 2, 1 } };

    Backward(errorLayer2, connection1to2, derivativeConnection1to2);

    assert(derivativeConnection1to2[0].weight == 10);
    assert(derivativeConnection1to2[1].weight == -10);
}

void RunTests()
{
    // Matrix input {10, 100, 100};
    // Matrix connection {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    // Matrix output = input * connection;

    ML_Matrix<float> layer1{ Int2{ 3, 1 }, {10, 100, 1000} };

    ML_Matrix<float> layer2{ Int2{ 4, 1 } };

    ML_Matrix<float> connection1to2{ ML_DenseConnection::ConnectionMatrixSize(layer1, layer2),
        {1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         1, 1, 1}};

    Multiply(layer1, connection1to2, layer2);

    // Verify that the result vector is correct
    assert(layer2[0] == 10);
    assert(layer2[1] == 100);
    assert(layer2[2] == 1000);
    assert(layer2[3] == 1110);

    printf("Test PASSED\n");


    ML_Matrix<float> derivative1{ Int2{ 4, 1 } };
    Divide(layer1, connection1to2, derivative1);
    //Divide(layer1, connection1to2, derivative1);

    assert(derivative1[0] == 10);
    assert(derivative1[1] == 100);
    assert(derivative1[2] == 1000);
    assert(derivative1[3] == 1110);
}

/**
 * Host main routine
 */
int main(void) {

    RunTests();
    RunNetwork();

    printf("Done\n");
    return 0;
}
