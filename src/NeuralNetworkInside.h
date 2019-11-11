//
// Created by emile on 02/08/2019.
//

#ifndef NEURAL_NETWORK_NEURALNETWORKINSIDE_H
#define NEURAL_NETWORK_NEURALNETWORKINSIDE_H

#include <iostream>
#include <array>
#include <variant>
#include <typeinfo>
#include <random>
#include <cmath>
#include <cassert>
#include "Matrix.h"
#include "relevant_math.h"
#include "RecTuple.h"
#include "DeltaNetwork.h"
#include "NeuralNetworkBase.h"

//using namespace std;

template<size_t ...sizes>
class NeuralNetworkInside final
{

};

template<size_t size1, size_t size2>
class NeuralNetworkInside<size1, size2> : public NeuralNetworkBase<NeuralNetworkInside, size1, size2>
{

//region constexpr static

public:

    using base =  NeuralNetworkBase<NeuralNetworkInside, size1, size2>;
    using input_t = typename base::input_t;
    using output_t = typename base::output_t;

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }

//endregion

//region constructors

public:

    NeuralNetworkInside(DMatrix<size2, size1> weights_, DVector<size1> biases_) :
            NeuralNetworkBase<NeuralNetworkInside, size1, size2>(weights_, biases_)
    {}

    NeuralNetworkInside() : NeuralNetworkBase<NeuralNetworkInside, size1, size2>()
    {}

//endregion

//region methods

    DVector<size1> inner_predict(const DVector<size2> &input) const
    {
        return (this->weights * input + this->biases).fmap(function(sigmoid));
    }

    RecTuple<DVector<size1>, DVector<size2>> inner_feedforward(const DVector<size2> &input) const
    {
        return {inner_predict(input), RecTuple<DVector<size2>>(input)};
    }

//endregion

};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkInside<size1, size2, sizes...>
        : public NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>
{

//region constexpr static

    using base =  NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>;
    using other_layers_t = NeuralNetworkInside<size2, sizes...>;

    constexpr static size_t get_number_of_layers()
    {
        return 1 + NeuralNetworkInside<size2, sizes...>::get_number_of_layers();
    }

    using input_t = DVector<base::last_size()>;
    using output_t = DVector<size1>;

//endregion

//region constructors

public:

    NeuralNetworkInside(DMatrix<size2, size1> weights_, DVector<size1> biases_, other_layers_t other_layers_) :
            NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>(weights_, biases_, other_layers_)
    {}

    NeuralNetworkInside() : NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>()
    {}
//endregion

//region methods

    DVector<size1> apply_one_iter(const DVector<size2> &m) const
    {
        return (this->weights * m + this->biases).fmap(function(sigmoid));
    }

    RecTuple<DVector<size1>, DVector<size2>, DVector<sizes>...>
    inner_feedforward(const DVector<base::last_size()> &input) const
    {
        RecTuple<DVector<size2>, DVector<sizes>...>
                semi_result = this->other_layers.inner_feedforward(input);
        auto r = apply_one_iter(semi_result.head);

        //if constexpr (tuple_size<)
        void (*signal(int sig, void (*func)(int)))(int);

        return {r, semi_result};

    }

    DVector<size1> inner_predict(const DVector<base::last_size()> &input) const
    {
        DVector<size2> semi_result = this->other_layers.inner_predict(input);
        return apply_one_iter(semi_result);
    }

//endregion

};


#endif //NEURAL_NETWORK_NEURALNETWORKINSIDE_H
