//
// Created by emile on 02/08/2019.
//

#ifndef NEURAL_NETWORK_DLAYER_H
#define NEURAL_NETWORK_DLAYER_H

#include <iostream>
#include <array>
#include <variant>
#include <typeinfo>
#include <tuple>
#include <random>
#include <cmath>
#include <cassert>
#include "Matrix.h"
#include "relevant_math.h"
#include "RecTuple.h"

//region class D_Layer

template<size_t size1, size_t size2>
class D_Layer
{
public:
    Matrix<double, size2, size1> d_weights;
    Vector<double, size2> d_input;
    Vector<double, size1> d_biases;

    D_Layer() : d_weights(MatrixFactory::uniform<double, size2, size1>(0)),
                d_input(MatrixFactory::uniform<double, 1, size2>(0)),
                d_biases(MatrixFactory::uniform<double, 1, size1>(0))
    {
    }

    D_Layer(Matrix<double, size2, size1> d_weights_,
            Vector<double, size2> d_input_,
            Vector<double, size2> d_biases_) : d_weights(d_weights_),
                                               d_biases(d_biases_),
                                               d_input(d_input_)
    {

    }

    D_Layer<size1, size2> operator+(D_Layer<size1, size2> other) const
    {
        return D_Layer(d_weights + other.d_weights, d_input + other.d_input, d_biases + other.d_biases);
    }

    D_Layer<size1, size2> operator+=(D_Layer<size1, size2> other)
    {
        d_weights += other.d_weights;
        d_biases += other.d_biases;
        d_input += other.d_input;
        return *this;
    }

    D_Layer<size1, size2> operator*(double x) const
    {
        return D_Layer(d_weights * x, d_input * x, d_biases * x);
    }

    D_Layer<size1, size2> &operator*=(double x)
    {
        d_weights *= x;
        d_input *= x;
        d_biases *= x;
        return *this;
    }

};

//endregion



#endif //NEURAL_NETWORK_DLAYER_H
