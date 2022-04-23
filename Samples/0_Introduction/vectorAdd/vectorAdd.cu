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

__global__ void vectorMultiply(const ML_DeviceMatrix<float> A, const ML_DeviceMatrix<float> B, ML_DeviceMatrix<float> C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Int2::Size(C.dimensions)) 
    {
        float* bufferRootB = &B.deviceBuffer[i * A.dimensions.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.dimensions.x; elementIndex++)
        {
            total += bufferRootB[elementIndex] * A.deviceBuffer[elementIndex];
        }

        C.deviceBuffer[i] = total;
    }
}

__global__ void vectorDivide(const ML_DeviceMatrix<float> A, const ML_DeviceMatrix<float> B, ML_DeviceMatrix<float> C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Int2::Size(C.dimensions))
    {
        float* bufferRootB = &B.deviceBuffer[i * A.dimensions.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.dimensions.x; elementIndex++)
        {
            float b = bufferRootB[elementIndex];
            if (b != 0)
            {
                total += A.deviceBuffer[elementIndex] / b;
            }
        }

        C.deviceBuffer[i] = total;
    }
}

void Multiply(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output)
{
    ML_Helpers::VerifyForwardConnection(input.Dimensions(), connection.Dimensions(), output.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ output.Dimensions()};
	vectorMultiply CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), connection.DeviceArray(), output.DeviceArray());
}

void Divide(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output)
{
    ML_Helpers::VerifyForwardConnection(input.Dimensions(), connection.Dimensions(), output.Dimensions());
    ML_CheckCudaError checkError;
    ML_KernelSize size{ output.Dimensions()};
    vectorDivide CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.DeviceArray(), connection.DeviceArray(), output.DeviceArray());
}

void Run()
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

    assert(derivative1[0] == 10);
    assert(derivative1[1] == 100);
    assert(derivative1[2] == 1000);
    assert(derivative1[3] == 1110);
}

/**
 * Host main routine
 */
int main(void) {

    Run();

    printf("Done\n");
    return 0;
}
