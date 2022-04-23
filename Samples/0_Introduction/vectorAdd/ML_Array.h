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
    Int2 dimensions;
};

template <class Type>
struct ML_DeviceMatrixAllocation
{
    ML_DeviceMatrixAllocation(Int2 dimensions)
    {
        matrix.dimensions = dimensions;

        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        // Allocate the device input vector A
        err = cudaMalloc((void**)&matrix.deviceBuffer, AllocationSize(dimensions));
        
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
        err = cudaFree(matrix.deviceBuffer);

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    static size_t AllocationSize(const Int2 dimensions)
    {
        return dimensions.Size() * sizeof(Type);
    }

    void HostToDevice(Type* hostBuffer)
    {
        cudaError_t err = cudaSuccess;

        // Copy the host input vectors A and B in host memory to the device input vectors in device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(matrix.deviceBuffer, hostBuffer, AllocationSize(matrix.dimensions), cudaMemcpyHostToDevice);

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

        // Copy the device result vector in device memory to the host result vector in host memory.
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(hostBuffer, matrix.deviceBuffer, AllocationSize(matrix.dimensions), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    ML_DeviceMatrix<Type> matrix;
};

enum class ML_SyncState : uint8_t
{
    HostAuthorative,
    DeviceAuthorative,
};

template <class Type>
struct ML_Matrix
{
    ML_Matrix(Int2 dimensions)
        : deviceAllocation(dimensions)
    {
        // Allocate the host input vector A
        hostArray.resize(dimensions.Size());
        hostBuffer = &hostArray[0];

        // Verify that allocations succeeded
        if (hostBuffer == NULL) 
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }
    }

    ML_Matrix(Int2 dimensions, std::vector<Type>&& constructFrom)
        : deviceAllocation(dimensions)
    {
        // Allocate the host input vector A
        hostArray = constructFrom;
        assert(hostArray.size() == dimensions.Size());
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

    Int2 Dimensions() const
    {
        return deviceAllocation.matrix.dimensions;
    }

    int Index(Int2 position) const
    {
        return position.x + position.y * deviceAllocation.dimensions.x;
    }

    ML_DeviceMatrix<Type>& DeviceArray()
    {
        if (syncState == ML_SyncState::HostAuthorative)
        {
            syncState = ML_SyncState::DeviceAuthorative;
            deviceAllocation.HostToDevice(hostBuffer);
        }
        return deviceAllocation.matrix;
    }

    std::vector<Type>& HostArray()
    {
        if (syncState == ML_SyncState::DeviceAuthorative)
        {
            syncState = ML_SyncState::HostAuthorative;
            deviceAllocation.DeviceToHost(hostBuffer);
        }
        return hostArray;
    }

    Type& operator[] (int index)
    {
        return HostArray()[index];
    }
    Type& operator[] (Int2 position)
    {
        return this[Index(position)];
    }
private:
    // Host
    std::vector<Type> hostArray;
    Type* hostBuffer;
    ML_DeviceMatrixAllocation<Type> deviceAllocation;
    ML_SyncState syncState = ML_SyncState::HostAuthorative;
};