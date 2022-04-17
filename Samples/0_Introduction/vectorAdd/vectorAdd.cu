#include <stdio.h>
#include <vector>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
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

struct Int2
{
    int x;
    int y;

    int Size()
    {
        return x * y;
    }
};

struct ML_Array
{
    ML_Array(Int2 numElements)
        : numElements(numElements)
        , size(numElements.Size() * sizeof(float))
    {
        // Allocate the host input vector A
        hostArray.resize(numElements.Size());
        hostBuffer = &hostArray[0];

        //h_A = (float*)malloc(size);


        // Verify that allocations succeeded
        if (hostBuffer == NULL) {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&deviceBuffer, size);

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    ~ML_Array()
    {
        cudaError_t err = cudaSuccess;

        // Free device global memory
        err = cudaFree(deviceBuffer);

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Free host memory
        //free(h_A);
    }

    void InitializeToRandomValues()
    {
        for (int i = 0; i < numElements.Size(); ++i)
        {
            hostBuffer[i] = rand() / (float)RAND_MAX;
        }
    }

    void HostToDevice()
    {
        cudaError_t err = cudaSuccess;

        // Copy the host input vectors A and B in host memory to the device input
        // vectors in
        // device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(deviceBuffer, hostBuffer, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void DeviceToHost()
    {
        cudaError_t err = cudaSuccess;

        // Copy the device result vector in device memory to the host result vector
// in host memory.
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(hostBuffer, deviceBuffer, size, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Host
    std::vector<float> hostArray;
    float* hostBuffer;
    // Device
    float* deviceBuffer;

    size_t size;
    Int2 numElements;
};

void Run()
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    Int2 numElements{ 50000, 1 };
    size_t size = numElements.Size() * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements.Size());

    ML_Array arrayA{ numElements };
    arrayA.InitializeToRandomValues();

	ML_Array arrayB{ numElements };
    arrayB.InitializeToRandomValues();

	ML_Array arrayC{ numElements };

    arrayA.HostToDevice();
    arrayB.HostToDevice();

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements.Size() + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
        threadsPerBlock);
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (arrayA.deviceBuffer, arrayB.deviceBuffer, arrayC.deviceBuffer, numElements.Size());
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    arrayC.DeviceToHost();

    // Verify that the result vector is correct
    for (int i = 0; i < numElements.Size(); ++i) {
        if (fabs(arrayA.hostBuffer[i] + arrayB.hostBuffer[i] - arrayC.hostBuffer[i]) > 1e-5) {
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
