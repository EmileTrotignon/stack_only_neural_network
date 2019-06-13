//
// Created by emile on 11/06/19.
//

#ifndef NEURAL_NETWORK_RELEVANT_MATH_H
#define NEURAL_NETWORK_RELEVANT_MATH_H

#include "Matrix.h"

double sigmoid(double input);

double sigmoid_derv(double sigmoid_x);

template<size_t H>
Matrix<double, 1, H> softmax(Matrix<double, 1, H> m)
{
    double max = m.max();
    auto r = m.fmap(function([=](double x)
                             { return exp(x - max); }));
    double sum = r.sum();
    return r.fmap(function([=](double x)
                           { return x / sum; }));

}

#endif //NEURAL_NETWORK_RELEVANT_MATH_H
