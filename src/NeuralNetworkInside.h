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
class NeuralNetworkInside
{

public:

    NeuralNetworkInside() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetworkInside<size1, size2> : public NeuralNetworkBase<size1, size2>
{

//region constexpr static

    using base =  NeuralNetworkBase<size1, size2>;

public:
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

    NeuralNetworkInside() : base()
    {
    }

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

    DeltaLayer<size1, size2>
    compute_layer_changes(const DVector<size2> input,
                          const DVector<size1> prediction,
                          const DVector<size1> expected_result) const
    {

        DVector<size1> a_delta = DVector<size1>::element_by_element_product(
                (expected_result - prediction), prediction.fmap(function(sigmoid_derv)));

        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_biases = a_delta;
        return d_layer;


    }

    DeltaNetwork<size1, size2>
    inner_backpropagate_one_input(const RecTuple<DVector<size1>,
            DVector<size2>> &predictions,
                                  const DVector<size1> &expected_output) const
    {
        auto a0 = (predictions.tail.head);
        auto a1 = (predictions.head);
        DeltaLayer d_layer = compute_layer_changes(a0, a1, expected_output);
        return DeltaNetwork<size1, size2>(d_layer);

    }

//endregion

};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkInside<size1, size2, sizes...> : public NeuralNetworkBase<size1, size2, sizes...>
{

//region constexpr static

    using base =  NeuralNetworkBase<size1, size2, sizes...>;

    constexpr static size_t get_number_of_layers()
    {
        return 1 + NeuralNetworkInside<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if (i == 0)
        {
            return size1;
        } else
        {
            return NeuralNetworkInside<size2, sizes...>::template NeuralNetworkInside<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return NeuralNetworkInside<size2, sizes...>::last_size();
    }

//endregion

//region constructors

public:
    NeuralNetworkInside() : base()
    {
    }
//endregion

//region methods

    DVector<size1> apply_one_iter(const DVector<size2> &m) const
    {
        return (this->weights * m + this->biases).fmap(function(sigmoid));
    }

    RecTuple<DVector<size1>, DVector<size2>, DVector<sizes>...>
    inner_feedforward(const DVector<last_size()> &input) const
    {
        RecTuple<DVector<size2>, DVector<sizes>...>
                semi_result = this->other_layers.inner_feedforward(input);
        auto r = apply_one_iter(semi_result.head);

        //if constexpr (tuple_size<)
        return {r, semi_result};
    }

    DVector<size1> inner_predict(const DVector<last_size()> &input) const
    {
        DVector<size2> semi_result = this->other_layers.inner_predict(input);
        return apply_one_iter(semi_result);
    }

    DeltaLayer<size1, size2>
    compute_layer_changes(const DVector<size2> &input,
                          const DVector<size1> &prediction,
                          const DVector<size1> &expected_result) const
    {
        DVector<size1> a_delta = DVector<size1>::element_by_element_product(
                (expected_result - prediction), prediction.fmap(function(sigmoid_derv)));

        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_input = this->weights.transpose() * a_delta;
        d_layer.d_biases = a_delta;
        return d_layer;


    }

    DeltaNetwork<size1, size2, sizes...>
    inner_backpropagate_one_input(const RecTuple<DVector<size1>,
            DVector<size2>,
            DVector<sizes>...> &predictions,
                                  const DVector<size1> &expected_output) const
    {
        DeltaLayer d_layer = compute_layer_changes(predictions.tail.head, predictions.head, expected_output);

        DeltaNetwork<size2, sizes...> other_changes = this->other_layers.inner_backpropagate_one_input(predictions.tail,
                                                                                                       (predictions.tail.head) +
                                                                                              d_layer.d_input);

        return DeltaNetwork<size1, size2, sizes...>(d_layer, other_changes);

    }
//endregion
};


#endif //NEURAL_NETWORK_NEURALNETWORKINSIDE_H
