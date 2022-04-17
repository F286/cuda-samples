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
        h_A = &hostArray[0];

        //h_A = (float*)malloc(size);

		for (int i = 0; i < numElements.Size(); ++i) 
        {
			h_A[i] = rand() / (float)RAND_MAX;
		}

        // Verify that allocations succeeded
        if (h_A == NULL) {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&d_A, size);

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
        err = cudaFree(d_A);

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Free host memory
        //free(h_A);
    }

    void HostToDevice()
    {
        cudaError_t err = cudaSuccess;

        // Copy the host input vectors A and B in host memory to the device input
        // vectors in
        // device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    // Host
    std::vector<float> hostArray;
    float* h_A;
    // Device
    float* d_A;

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

    // Allocate the host input vector A
    //float *h_A = (float *)malloc(size);
    ML_Array arrayA{ numElements };

    // Allocate the host input vector B
    //float* h_B = (float*)malloc(size);

    // Allocate the host output vector C
    float* h_C = (float*)malloc(size);

    // Verify that allocations succeeded
    if (h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements.Size(); ++i) {
        //h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device input vector A
    //float *d_A = NULL;
    //err = cudaMalloc((void **)&d_A, size);

    //if (err != cudaSuccess) {
    //  fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
    //          cudaGetErrorString(err));
    //  exit(EXIT_FAILURE);
    //}

    // Allocate the device input vector B
    float* d_B = NULL;
    err = cudaMalloc((void**)&d_B, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float* d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //// Copy the host input vectors A and B in host memory to the device input
    //// vectors in
    //// device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    //err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    arrayA.HostToDevice();

    //if (err != cudaSuccess) {
    //  fprintf(stderr,
    //          "Failed to copy vector A from host to device (error code %s)!\n",
    //          cudaGetErrorString(err));
    //  exit(EXIT_FAILURE);
    //}

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements.Size() + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
        threadsPerBlock);
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (arrayA.d_A, d_B, d_C, numElements.Size());
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements.Size(); ++i) {
        if (fabs(arrayA.h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    //// Free device global memory
    //err = cudaFree(d_A);

    //if (err != cudaSuccess) {
    //  fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
    //          cudaGetErrorString(err));
    //  exit(EXIT_FAILURE);
    //}

    err = cudaFree(d_B);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    //free(h_A);
    free(h_B);
    free(h_C);
}

/**
 * Host main routine
 */
int main(void) {

    Run();

    printf("Done\n");
    return 0;
}
