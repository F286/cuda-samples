#include <stdio.h>
#include <vector>
#include <assert.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include "ML_Array.h"

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

__global__ void vectorMultiply(const ML_DeviceArray A, const ML_DeviceArray B, ML_DeviceArray C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < C.numElements.x * C.numElements.y) 
    {
        float* bufferRootB = &B.deviceBuffer[i * A.numElements.x];

        float total = 0.0f;
        for (int elementIndex = 0; elementIndex < A.numElements.x; elementIndex++)
        {
            total += A.deviceBuffer[elementIndex] * bufferRootB[elementIndex];
        }

        C.deviceBuffer[i] = total;
        //C.deviceBuffer[i] = A.deviceBuffer[i] * B.deviceBuffer[i] + 0.0f;
    }
}

void Multiply(ML_Array& arrayA, ML_Array& arrayB, ML_Array& arrayOut)
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
    vectorMultiply << <blocksPerGrid, threadsPerBlock >> > (arrayA.deviceArray, arrayB.deviceArray, arrayOut.deviceArray);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Run()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    ML_Array arrayA{ Int2{3, 1} };
    //arrayA.InitializeToRandomValues();
    arrayA[Int2{ 0, 0 }] = 10;
    arrayA[Int2{ 1, 0 }] = 100;
    arrayA[Int2{ 2, 0 }] = 1000;

    //ML_Array arrayB{ Int2{3, 1} };
    ML_Array arrayB{ Int2{3, 4} };
    //arrayB.InitializeToRandomValues();
    arrayB[Int2{ 0, 0 }] = 1;
    arrayB[Int2{ 1, 0 }] = 0;
    arrayB[Int2{ 2, 0 }] = 0;

    arrayB[Int2{ 0, 1 }] = 0;
    arrayB[Int2{ 1, 1 }] = 1;
    arrayB[Int2{ 2, 1 }] = 0;

    arrayB[Int2{ 0, 2 }] = 0;
    arrayB[Int2{ 1, 2 }] = 0;
    arrayB[Int2{ 2, 2 }] = 1;

    arrayB[Int2{ 0, 3 }] = 1;
    arrayB[Int2{ 1, 3 }] = 1;
    arrayB[Int2{ 2, 3 }] = 1;

    ML_Array arrayC{ Int2{4, 1} };

    arrayA.HostToDevice();
    arrayB.HostToDevice();

    // Launch the Vector Multiply CUDA Kernel
    Multiply(arrayA, arrayB, arrayC);

    arrayC.DeviceToHost();

    // Verify that the result vector is correct
    for (int i = 0; i < arrayC.NumElements().Size(); ++i) {
        if (fabs(arrayA.hostBuffer[i] * arrayB.hostBuffer[i] - arrayC.hostBuffer[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

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
