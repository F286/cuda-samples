#include <stdio.h>
#include <vector>
#include <assert.h>
#include <memory>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include "ML_Array.h"

struct ML_DenseConnection;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

template <class ConnectionType>
__global__ void vectorMultiply(const ML_DeviceArray<float> A, const ML_DeviceArray<float> B, ML_DeviceArray<float> C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Int2::Size(C.numElements)) 
    {
        float* bufferRootB = &B.deviceBuffer[i * A.numElements.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.numElements.x; elementIndex++)
        {
            total += ConnectionType::Run(A.deviceBuffer[elementIndex], bufferRootB[elementIndex]);
        }

        C.deviceBuffer[i] = total;
    }
}

// Possible to pass in method to execute on cuda vectors with template using global method?
void Multiply(ML_Array<float>& arrayA, ML_Array<float>& arrayB, ML_Array<float>& arrayOut)
{
    assert(arrayB.NumElements().x == arrayA.NumElements().x);
    assert(arrayB.NumElements().y == arrayOut.NumElements().x);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    Int2 numElements = arrayOut.deviceArray.numElements;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements.Size() + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
        threadsPerBlock);
    vectorMultiply<ML_DenseConnection> << <blocksPerGrid, threadsPerBlock >> > (arrayA.deviceArray, arrayB.deviceArray, arrayOut.deviceArray);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

struct ML_DenseConnection
{
    ML_DenseConnection(ML_Array<float>& previous, ML_Array<float>& next)
        : previous(previous)
        , next(next)
        , connection(ConnectionMatrixSize(previous, next))
    {
    }

    void Run()
    {
        previous.HostToDevice();
        connection.HostToDevice();

        // Launch the Vector Multiply CUDA Kernel
        Multiply(previous, connection, next);

        next.DeviceToHost();
    }

    static Int2 ConnectionMatrixSize(const ML_Array<float>& previous, const ML_Array<float>& next)
    {
        assert(previous.NumElements().y == 1);
        assert(next.NumElements().y == 1);
        return Int2{ previous.NumElements().x, next.NumElements().x };
    }

    __host__ __device__ static float Run(const float& previous, const float& connection)
    {
        return previous * connection;
    }

    float& operator[] (int index)
    {
        return connection[index];
    }
    float& operator[] (Int2 position)
    {
        return connection[position];
    }

    ML_Array<float>& previous;
    ML_Array<float>& next;

    ML_Array<float> connection;
};

struct ML_ValueDecorator
{
};
struct ML_ConnectionDecorator
{
};

struct ML_DenseArrayDerivative : public ML_ValueDecorator
{
    // TODO (fd) : ML_Array should be an object that supports weights and biases. Struct packed together for memory access efficiency.

    ML_Array<float>& original;
    ML_Array<float> derivative;
};

struct ML_DenseConnectionDerivative : public ML_ConnectionDecorator
{

};

void Run()
{
    ML_Array<float> array1{ Int2{ 3, 1 } };
    array1[Int2{ 0, 0 }] = 10;
    array1[Int2{ 1, 0 }] = 100;
    array1[Int2{ 2, 0 }] = 1000;

    ML_Array<float> array2{ Int2{ 4, 1 } };

	ML_DenseConnection connection1{ array1, array2 };

    connection1[Int2{ 0, 0 }] = 1;
    connection1[Int2{ 1, 0 }] = 0;
    connection1[Int2{ 2, 0 }] = 0;

    connection1[Int2{ 0, 1 }] = 0;
    connection1[Int2{ 1, 1 }] = 1;
    connection1[Int2{ 2, 1 }] = 0;

    connection1[Int2{ 0, 2 }] = 0;
    connection1[Int2{ 1, 2 }] = 0;
    connection1[Int2{ 2, 2 }] = 1;

    connection1[Int2{ 0, 3 }] = 1;
    connection1[Int2{ 1, 3 }] = 1;
    connection1[Int2{ 2, 3 }] = 1;

    connection1.Run();

    // Verify that the result vector is correct
    assert(array2[0] == 10);
    assert(array2[1] == 100);
    assert(array2[2] == 1000);
    assert(array2[3] == 1110);

    printf("Test PASSED\n");
}

/**
 * Host main routine
 */
int main(void) {

    Run();

    printf("Done\n");
    return 0;
}
