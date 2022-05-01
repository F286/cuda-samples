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

void RandomizeWeights(ML_Matrix<ML_Neuron>& connection)
{
    for (int i = 0; i < connection.Dimensions().Count(); i++)
    {
        connection[i] = ML_Neuron{ 0.5f + (i % 5) * 0.1f, 0.5f + (i % 5) * 0.1f };
        //connection[i] = ML_Neuron{ rand() / (float)RAND_MAX, rand() / (float)RAND_MAX };
    }
}
float Loss(ML_Matrix<float>& values)
{
    float total = 0.0f;

    for (int i = 0; i < values.Dimensions().Count(); i++)
    {
        total += abs(values[i]);
    }

    return total;
}

void RunNetwork()
{
    const int INPUT_COUNT = 1;
    const int CONNECTION_COUNT = 1;
    const int OUTPUT_COUNT = 2;
    const int TRAINING_STEPS = 100;
    //const float TRAINING_RATE = 0.5f;
    const float TRAINING_RATE = 0.1f;


    ML_Matrix<float> value1{ Int2{ INPUT_COUNT, 1 }, {10} };
    //ML_Matrix<float> value1{ Int2{ INPUT_COUNT, 1 }, {10, -5} };
    ML_Matrix<float> value2{ Int2{ CONNECTION_COUNT, 1 } };
    ML_Matrix<float> value3{ Int2{ OUTPUT_COUNT, 1 } };
    ML_Matrix<ML_Neuron> connection1_2{ Int2{ INPUT_COUNT, CONNECTION_COUNT } };
    ML_Matrix<ML_Neuron> connection2_3{ Int2{ CONNECTION_COUNT, OUTPUT_COUNT } };
    //ML_Matrix<ML_Neuron> connection1_3{ Int2{ INPUT_COUNT, OUTPUT_COUNT } };

    ML_Matrix<float> expected3{ value3.Dimensions(), { 10, 2 } };
    ML_Matrix<float> error3{ value3.Dimensions() };

    ML_Matrix<ML_Neuron> derivative1_2{ connection1_2.Dimensions() };
    ML_Matrix<ML_Neuron> derivative2_3{ connection2_3.Dimensions() };
    //ML_Matrix<ML_Neuron> derivative1_3{ connection1_3.Dimensions() };
    ML_Matrix<float> derivative1{ value1.Dimensions() };
    ML_Matrix<float> derivative2{ value2.Dimensions() };

    // Randomize weights
    RandomizeWeights(connection1_2);
    RandomizeWeights(connection2_3);
    //RandomizeWeights(connection1_3);

    for (int i = 0; i < TRAINING_STEPS; i++)
    {
        // Run forward
        Forward(value1, connection1_2, value2);
        Forward(value2, connection2_3, value3);
        //Forward(value1, connection1_3, value3);

        // Calculate error
        Error(value3, expected3, error3);

        // Derivative
        derivative1.Clear();
        derivative2.Clear();
        Backward(error3, connection2_3, value2, derivative2_3, derivative2);
        Backward(derivative2, connection1_2, value1, derivative1_2, derivative1);
        //Backward(error3, connection1_3, value1, derivative1_3, derivative1);

        // Apply training
        Apply(connection2_3, derivative2_3, TRAINING_RATE);
        Apply(connection1_2, derivative1_2, TRAINING_RATE);
        //Apply(connection1_3, derivative1_3, TRAINING_RATE);

        // Print
        const float loss = Loss(error3);
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
