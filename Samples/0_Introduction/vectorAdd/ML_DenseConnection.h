#pragma once
#include "ML_Array.h"
#include <stdio.h>

namespace ML_DenseConnection
{
    static Int2 ConnectionMatrixSize(const ML_Matrix<float>& previous, const ML_Matrix<float>& next)
    {
        assert(previous.NumElements().y == 1);
        assert(next.NumElements().y == 1);
        return Int2{ previous.NumElements().x, next.NumElements().x };
    }
}
