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

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//__global__ void vectorAdd(const float *A, const float *B, float *C,
//                          int numElements) {
//  int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//  if (i < numElements) {
//    C[i] = A[i] + B[i] + 0.0f;
//  }
//}

__global__ void vectorMultiply(const ML_DeviceMatrix<float> A, const ML_DeviceMatrix<float> B, ML_DeviceMatrix<float> C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Int2::Size(C.numElements)) 
    {
        float* bufferRootB = &B.deviceBuffer[i * A.numElements.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.numElements.x; elementIndex++)
        {
            total += bufferRootB[elementIndex] * A.deviceBuffer[elementIndex];
        }

        C.deviceBuffer[i] = total;
    }
}

__global__ void vectorDivide(const ML_DeviceMatrix<float> A, const ML_DeviceMatrix<float> B, ML_DeviceMatrix<float> C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Int2::Size(C.numElements))
    {
        float* bufferRootB = &B.deviceBuffer[i * A.numElements.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.numElements.x; elementIndex++)
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
    input.HostToDevice();
    connection.HostToDevice();

    assert(connection.NumElements().x == input.NumElements().x);
    assert(connection.NumElements().y == output.NumElements().x);

    ML_CheckCudaError checkError;

    ML_KernelSize size{ output.deviceArray.numElements };
	vectorMultiply CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.deviceArray, connection.deviceArray, output.deviceArray);
        
    output.DeviceToHost();
}

void Divide(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output)
{
    input.HostToDevice();
    connection.HostToDevice();

    assert(connection.NumElements().x == input.NumElements().x);
    assert(connection.NumElements().y == output.NumElements().x);

    ML_CheckCudaError checkError;

    ML_KernelSize size{ output.deviceArray.numElements };
    vectorDivide CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock)(input.deviceArray, connection.deviceArray, output.deviceArray);

    output.DeviceToHost();
}

void Run()
{
    // Matrix input {10, 100, 100};
    // Matrix connection {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    // Matrix output = input * connection;

    ML_Matrix<float> array1{ Int2{ 3, 1 }, {10, 100, 1000} };

    ML_Matrix<float> array2{ Int2{ 4, 1 } };

    ML_Matrix<float> connection1{ ML_DenseConnection::ConnectionMatrixSize(array1, array2),
        {1, 0, 0,
         0, 1, 0,
         0, 0, 1,
         1, 1, 1}};

    Multiply(array1, connection1, array2);

    // Verify that the result vector is correct
    assert(array2[0] == 10);
    assert(array2[1] == 100);
    assert(array2[2] == 1000);
    assert(array2[3] == 1110);

    printf("Test PASSED\n");


    ML_Matrix<float> array3{ Int2{ 4, 1 } };
    Divide(array1, connection1, array3);

    assert(array3[0] == 10);
    assert(array3[1] == 100);
    assert(array3[2] == 1000);
    assert(array3[3] == 1110);
}

/**
 * Host main routine
 */
int main(void) {

    Run();

    printf("Done\n");
    return 0;
}
