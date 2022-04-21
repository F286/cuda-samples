#pragma once
#include <stdio.h>
#include <vector>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <cuda_runtime.h>

struct Int2
{
    int x;
    int y;
 
    int Size() const
    {
        return Int2::Size(*this);
    }

    __host__ __device__ static int Size(const Int2& instance)
    {
        return instance.x * instance.y;
    }
};

template <class Type>
struct ML_DeviceMatrix
{
    // Device
    Type* deviceBuffer;
    Int2 numElements;

    static size_t AllocationSize(const Int2 numElements)
    {
        return numElements.Size() * sizeof(Type);
    }
};

template <class Type>
struct ML_DeviceMatrixAllocation : public ML_DeviceMatrix<Type>
{
    ML_DeviceMatrixAllocation(Int2 numElements)
    {
        this->numElements = numElements;

        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&deviceBuffer, ML_DeviceMatrix<Type>::AllocationSize(numElements));
        
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    ML_DeviceMatrixAllocation(const ML_DeviceMatrixAllocation&) = delete;

    ~ML_DeviceMatrixAllocation()
    {
        cudaError_t err = cudaSuccess;

        // Free device global memory
        err = cudaFree(deviceBuffer);

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void HostToDevice(Type* hostBuffer)
    {
        cudaError_t err = cudaSuccess;

        // Copy the host input vectors A and B in host memory to the device input
        // vectors in
        // device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(deviceBuffer, hostBuffer, ML_DeviceMatrix<Type>::AllocationSize(numElements), cudaMemcpyHostToDevice);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    void DeviceToHost(Type* hostBuffer)
    {
        cudaError_t err = cudaSuccess;

        // Copy the device result vector in device memory to the host result vector
// in host memory.
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(hostBuffer, deviceBuffer, ML_DeviceMatrix<Type>::AllocationSize(numElements), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
};

template <class Type>
struct ML_Matrix
{
    ML_Matrix(Int2 numElements)
        : deviceArray(numElements)
    {
        // Allocate the host input vector A
        hostArray.resize(numElements.Size());
        hostBuffer = &hostArray[0];

        // Verify that allocations succeeded
        if (hostBuffer == NULL) 
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }
    }

    ML_Matrix(Int2 numElements, std::vector<Type>&& constructFrom)
        : deviceArray(numElements)
    {
        // Allocate the host input vector A
        hostArray = constructFrom;
        assert(hostArray.size() == numElements.Size());
        hostBuffer = &hostArray[0];

        // Verify that allocations succeeded
        if (hostBuffer == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }
    }

    ML_Matrix(const ML_Matrix&) = delete;

    //void InitializeToRandomValues()
    //{
    //    for (int i = 0; i < hostArray.size(); ++i)
    //    {
    //        hostBuffer[i] = rand() / (float)RAND_MAX;
    //    }
    //}

    void HostToDevice()
    {
        deviceArray.HostToDevice(hostBuffer);
    }

    void DeviceToHost()
    {
        deviceArray.DeviceToHost(hostBuffer);
    }

    Int2 NumElements() const
    {
        return deviceArray.numElements;
    }

    int Index(Int2 position) const
    {
        return position.x + position.y * deviceArray.numElements.x;
    }

    Type& operator[] (int index)
    {
        return *(hostBuffer + index);
    }
    Type& operator[] (Int2 position)
    {
        return *(hostBuffer + Index(position));
    }

    // Host
    std::vector<Type> hostArray;
    Type* hostBuffer;
    ML_DeviceMatrixAllocation<Type> deviceArray;
};