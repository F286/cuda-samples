#pragma once
#include "ML_Array.h"
#include "ML_CheckCudaError.h"
#include "ML_DenseConnection.h"
#include "ML_Helpers.h"
#include "ML_Neuron.h"
#include "vectorMultiply.cuh"
#include "vectorNeuron.cuh"

void RunNetwork()
{
    ML_Matrix<float> layer1{ Int2{ 2, 1 }, {10, 100} };

    ML_Matrix<float> layer2{ Int2{ 2, 1 } };

    //ML_Matrix<ML_Neuron> connection1to2{ Int2{ 2, 1 }, { {0.1, 0}, {0.1, 2} } };
    ML_Matrix<ML_Neuron> connection1to2{ Int2{ 2, 2 }, 
        { {0.1, 0}, {0.1, 2}, 
          {0.1, 0}, {0.1, 2} } };

    // Run forward
    Forward(layer1, connection1to2, layer2);

    assert(layer2[0] == (1 + 12));
    assert(layer2[1] == (1 + 12));

    // Run back propagation
    ML_Matrix<float> errorLayer2{ layer2.Dimensions(), { 1, 1 } };
    ML_Matrix<ML_Neuron> derivativeConnection1to2{ connection1to2.Dimensions() };

    //Backward(errorLayer2, connection1to2, derivativeConnection1to2);

    ML_Matrix<float> errorLayer1{ layer1.Dimensions() };
    Backward(errorLayer2, connection1to2, derivativeConnection1to2, errorLayer1);

    assert(derivativeConnection1to2[0].weight == 10);
    assert(derivativeConnection1to2[1].weight == -10);
    assert(derivativeConnection1to2[2].weight == 10);
    assert(derivativeConnection1to2[3].weight == -10);

    assert(errorLayer1[0] == 20);
    assert(errorLayer1[1] == -20);
}

void RunTests()
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
         1, 1, 1}};

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

/**
 * Host main routine
 */
int main(void) {

    RunTests();
    RunNetwork();

    printf("Done\n");
    return 0;
}
