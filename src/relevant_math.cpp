//
// Created by emile on 11/06/19.
//

#include <cmath>
#include "relevant_math.h"


double sigmoid(double input)
{
    return 1 / (1 + exp(-input));
}



