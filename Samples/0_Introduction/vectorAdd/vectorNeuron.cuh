#pragma once

struct ML_Neuron;
template <class Type>
struct ML_Matrix;

void Forward(ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& output);
void Backward(ML_Matrix<float>& output, ML_Matrix<ML_Neuron>& connection, ML_Matrix<ML_Neuron>& connectionDerivative, ML_Matrix<float>& inputDerivative);
void Error(ML_Matrix<float>& value, ML_Matrix<float>& expected, ML_Matrix<float>& error);
//template<class Type>
//void Apply(ML_Matrix<Type>& original, ML_Matrix<Type>& deriviative, float rate);