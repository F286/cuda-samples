#pragma once

struct ML_Neuron;
template <class Type>
struct ML_Matrix;

void Forward(ML_Matrix<float>& input, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& output);
void Backward(ML_Matrix<float>& source, ML_Matrix<ML_Neuron>& connection, ML_Matrix<ML_Neuron>& derivative); 
void Backward(ML_Matrix<float>& source, ML_Matrix<ML_Neuron>& connection, ML_Matrix<float>& derivative);