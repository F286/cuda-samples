#pragma once
#include "ML_Helpers.h"

namespace ML_DenseConnection
{
    static Int2 ConnectionMatrixSize(const ML_Matrix<float>& previous, const ML_Matrix<float>& next)
    {
        assert(previous.Dimensions().y == 1);
        assert(next.Dimensions().y == 1);
        return Int2{ previous.Dimensions().x, next.Dimensions().x };
    }
}
