#pragma once
#include "ML_Array.h"
#include "ML_CheckCudaError.h"
#include "ML_DenseConnection.h"
#include "ML_Helpers.h"
#include "ML_Neuron.h"
#include "vectorMultiply.cuh"
#include "vectorNeuron.cuh"


template<class Type>
__global__ void vectorApply(ML_DeviceMatrix<Type> original, const ML_DeviceMatrix<Type> deriviative, const float rate) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < original.Count())
    {
        original[i] += deriviative[i] * rate;
    }
}
template<class Type>
void Apply(ML_Matrix<Type>& original, ML_Matrix<Type>& deriviative, float rate)
{
    assert(original.Dimensions() == deriviative.Dimensions());

    ML_CheckCudaError checkError;
    ML_KernelSize size{ original.Dimensions() };
    vectorApply CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock) (original.DeviceArray(), deriviative.DeviceArray(), rate);

    // Debug CPU copy back
    original.HostArray();
}

void RunNetwork()
{
    const int INPUT_COUNT = 2;
    const int CONNECTION_COUNT = 100;
    const int OUTPUT_COUNT = 2;

    ML_Matrix<float> value1{ Int2{ INPUT_COUNT, 1 }, {10, 100} };
    ML_Matrix<float> value2{ Int2{ OUTPUT_COUNT, 1 } };
    ML_Matrix<ML_Neuron> connection1_2{ Int2{ INPUT_COUNT, OUTPUT_COUNT } };

    ML_Matrix<float> expected2{ value2.Dimensions(), { 10, -10 } };
    ML_Matrix<float> error2{ value2.Dimensions(), { 1, 1 } };

    ML_Matrix<ML_Neuron> derivative1_2{ connection1_2.Dimensions() };
    ML_Matrix<float> derivative1{ value1.Dimensions() };

    // Randomize weights
    for (int i = 0; i < connection1_2.Dimensions().Count(); i++)
    {
        connection1_2[i] = ML_Neuron{ rand() / (float)RAND_MAX, rand() / (float)RAND_MAX };
    }

    for (int i = 0; i < 20; i++)
    {
        // Run forward
        Forward(value1, connection1_2, value2);

        // Calculate error
        Error(value2, expected2, error2);

        // Derivative
        Backward(error2, connection1_2, derivative1_2, derivative1);

        // Apply training
        Apply(connection1_2, derivative1_2, 0.0001f);

        // Print 
        float loss = abs(error2[0]) + abs(error2[1]);
        //printf("atomicSub failed\n");
        //error2[0];
        printf("Loss: %f\n", loss);
    }
}

void RunTests()
{
    // Basic multiply divide test
    {
        // Matrix input {10, 100, 100};
        // Matrix connection {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
        // Matrix output = input * connection;
        // 
        // Could probably do array size checking with static asserts at compile time?

        ML_Matrix<float> layer1{ Int2{ 3, 1 }, {10, 100, 1000} };

        ML_Matrix<float> layer2{ Int2{ 4, 1 } };

        ML_Matrix<float> connection1to2{ ML_DenseConnection::ConnectionMatrixSize(layer1, layer2),
            {1, 0, 0,
             0, 1, 0,
             0, 0, 1,
             1, 1, 1} };

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
