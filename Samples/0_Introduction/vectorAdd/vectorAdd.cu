#include <stdio.h>
#include <vector>

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

__global__ void vectorMultiply(const float* A, const float* B, float* C,
    int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] * B[i] + 0.0f;
    }
}

void Run()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    Int2 numElements{ 2, 1 };
    size_t size = numElements.Size() * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements.Size());

    ML_Array arrayA{ numElements };
    //arrayA.InitializeToRandomValues();
    arrayA[Int2{ 0, 0 }] = 2;
    arrayA[Int2{ 1, 0 }] = 3;

	ML_Array arrayB{ numElements };
    //arrayB.InitializeToRandomValues();
    arrayB[Int2{ 0, 0 }] = 5;
    arrayB[Int2{ 1, 0 }] = 6;

	ML_Array arrayC{ numElements };

    arrayA.HostToDevice();
    arrayB.HostToDevice();

    //// Launch the Vector Add CUDA Kernel
    //int threadsPerBlock = 256;
    //int blocksPerGrid = (numElements.Size() + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
    //    threadsPerBlock);
    //vectorAdd << <blocksPerGrid, threadsPerBlock >> > (arrayA.deviceBuffer, arrayB.deviceBuffer, arrayC.deviceBuffer, numElements.Size());
    //err = cudaGetLastError();

    //if (err != cudaSuccess) {
    //    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
    //        cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    
    // Launch the Vector Multiply CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements.Size() + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
        threadsPerBlock);
    vectorMultiply << <blocksPerGrid, threadsPerBlock >> > (arrayA.deviceBuffer, arrayB.deviceBuffer, arrayC.deviceBuffer, numElements.Size());
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    arrayC.DeviceToHost();

    //// Verify that the result vector is correct
    //for (int i = 0; i < numElements.Size(); ++i) {
    //    if (fabs(arrayA.hostBuffer[i] + arrayB.hostBuffer[i] - arrayC.hostBuffer[i]) > 1e-5) {
    //        fprintf(stderr, "Result verification failed at element %d!\n", i);
    //        exit(EXIT_FAILURE);
    //    }
    //}
    // Verify that the result vector is correct
    for (int i = 0; i < numElements.Size(); ++i) {
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
