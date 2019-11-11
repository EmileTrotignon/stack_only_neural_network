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

template<size_t size1, size_t size2>
class DeltaLayer
{
//region members
public:

    Matrix<double, size2, size1> d_weights;
    Vector<double, size2> d_input;
    Vector<double, size1> d_biases;

//endregion
//region constructors
public:

    DeltaLayer() : d_weights(Matrix<double, size2, size1>::uniform(0)),
                   d_input(Matrix<double, 1, size2>::uniform(0)),
                   d_biases(Matrix<double, 1, size1>::uniform(0))
    {
    }

    DeltaLayer(const Matrix<double, size2, size1> &d_weights_,
               const Vector<double, size2> &d_input_,
               const Vector<double, size1> &d_biases_) : d_weights(d_weights_),
                                                         d_biases(d_biases_),
                                                         d_input(d_input_)
    {

    }

    DeltaLayer(const Matrix<double, size2, size1> &d_weights_,
               const Vector<double, size1> &d_biases_) : d_weights(d_weights_),
                                                         d_biases(d_biases_),
                                                         d_input(Matrix<double, 1, size2>::uniform(0))
    {

    }
/*
    DeltaLayer(Matrix<double, size2, size1> &&d_weights_,
               Vector<double, size2> &&d_input_,
               Vector<double, size2> &&d_biases_) : d_weights(d_weights_),
                                                    d_biases(d_biases_),
                                                    d_input(d_input_)
    {

    }
*/
//endregion
//region operators
public :
    DeltaLayer<size1, size2> operator+(const DeltaLayer<size1, size2> &other) const
    {
        return DeltaLayer(d_weights + other.d_weights, d_input + other.d_input, d_biases + other.d_biases);
    }

    DeltaLayer<size1, size2> &operator+=(const DeltaLayer<size1, size2> &other)
    {
        d_weights += other.d_weights;
        d_biases += other.d_biases;
        d_input += other.d_input;
        return *this;
    }

    DeltaLayer<size1, size2> operator*(double x) const
    {
        return DeltaLayer(d_weights * x, d_input * x, d_biases * x);
    }

    DeltaLayer<size1, size2> &operator*=(double x)
    {
        d_weights *= x;
        d_input *= x;
        d_biases *= x;
        return *this;
    }
//endregion
};



#endif //NEURAL_NETWORK_DLAYER_H
