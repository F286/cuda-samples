#pragma once
#include "ML_Array.h"
#include "ML_CheckCudaError.h"
#include "ML_DenseConnection.h"
#include "ML_Helpers.h"
#include "ML_Neuron.h"
#include "vectorMultiply.cuh"
#include "vectorNeuron.cuh"

//template<class Type, class TRun>
//__global__ void vectorRun(const ML_DeviceMatrix<Type> a, const ML_DeviceMatrix<Type> b, ML_DeviceMatrix<Type> output)//, TRun run) {
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//
//    if (i < output.Count())
//    {
//        output[i] = TRun(a[i], b[i]);
//        //output[i] = run(a, b);
//        //output[i] = TRun(a, b);
//    }
//}
//template<class Type, class TRun>
//void Run(ML_Matrix<Type>& a, ML_Matrix<Type>& b, ML_Matrix<Type>& output)// TRun run)
//{
//    assert(a.Dimensions() == b.Dimensions());
//    assert(a.Dimensions() == output.Dimensions());
//
//    ML_CheckCudaError checkError;
//    ML_KernelSize size{ output.Dimensions() };
//    vectorRun<Type, TRun> CUDA_KERNEL(size.blocksPerGrid, size.threadsPerBlock) (a.DeviceArray(), b.DeviceArray(), output.DeviceArray());
//}

__device__ __host__ float CalculateError(float a, float b)
{
    return b - a;
}

void RunNetwork()
{
    ML_Matrix<float> value1{ Int2{ 2, 1 }, {10, 100} };
    ML_Matrix<float> value2{ Int2{ 2, 1 } };
    ML_Matrix<ML_Neuron> connection1_2{ Int2{ 2, 2 }, 
        { {0.1, 0}, {0.1, 2}, 
          {0.1, 0}, {0.1, 2} } };
    ML_Matrix<float> expected2{ value2.Dimensions(), { 10, -10 } };
    ML_Matrix<float> error2{ value2.Dimensions(), { 1, 1 } };
    ML_Matrix<ML_Neuron> derivative1_2{ connection1_2.Dimensions() };
    ML_Matrix<float> derivative1{ value1.Dimensions() };

    // Run forward
    Forward(value1, connection1_2, value2);

    assert(value2[0] == (1 + 12));
    assert(value2[1] == (1 + 12));

    // Calculate error
    Error(value2, expected2, error2);
    //Run<float, CalculateError>(value2, expected2, error2);// , [] __global__(float a, float b) -> float { return b - a; });

    assert(error2[0] != 0);

    // Run back propagation

    //Backward(errorLayer2, connection1to2, derivativeConnection1to2);
    //Error()

    Backward(error2, connection1_2, derivative1_2, derivative1);

    derivative1_2[0];
    derivative1[0];

    assert(derivative1_2[0].weight == 10);
    assert(derivative1_2[1].weight == -10);
    assert(derivative1_2[2].weight == 10);
    assert(derivative1_2[3].weight == -10);

    assert(derivative1[0] == 20);
    assert(derivative1[1] == -20);
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
