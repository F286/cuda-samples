#pragma once

struct ML_Neuron
{
    float weight;
    float bias;
    // Also uses ReLU

    __device__ __host__ ML_Neuron operator*(const ML_Neuron& other) const
    {
        return ML_Neuron{ weight * other.weight, bias * other.bias };
    };

    __device__ __host__ ML_Neuron operator*(const float other) const
    {
        return ML_Neuron{ weight * other, bias * other };
    };

    __device__ __host__ void operator+=(const ML_Neuron& other)
    {
        weight += other.weight;
        bias += other.bias;
    };
};

