#pragma once

template <class Type>
struct ML_Matrix;

void Multiply(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output);
void Divide(ML_Matrix<float>& input, ML_Matrix<float>& connection, ML_Matrix<float>& output);