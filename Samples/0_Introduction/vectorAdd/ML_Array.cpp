#include "ML_Array.h"

float& ML_Array::operator[] (int index)
{
    return *(hostBuffer + index);
}
float& ML_Array::operator[] (Int2 position)
{
    return *(hostBuffer + Index(position));
}