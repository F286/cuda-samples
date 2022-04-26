#include "ML_Helpers.h"

void ML_Helpers::VerifyForwardConnection(Int2 input, Int2 connection, Int2 output)
{
    assert(connection.x == input.x);
    assert(connection.y == output.x);
}

void ML_Helpers::VerifyBackwardConnection(Int2 source, Int2 connection, Int2 derivative)
{
    assert(connection == derivative);
    assert(source.x == connection.y);
}
