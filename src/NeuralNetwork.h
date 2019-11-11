//
// Created by emile on 07/06/19.
//

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include <iostream>
#include <array>
#include <variant>
#include <typeinfo>
#include <tuple>
#include <random>
#include <cmath>
#include <cassert>
#include <future>
#include <functional>
#include "Matrix.h"
#include "relevant_math.h"
#include "RecTuple.h"
#include "DeltaNetwork.h"
#include "NeuralNetworkBase.h"
#include "NeuralNetworkInside.h"

using namespace std;

//region NeuralNetwork

template<template<size_t...> typename NeuralNetworkInsideT, size_t ...sizes>
class NeuralNetwork final : public NeuralNetworkBase<NeuralNetworkInsideT, sizes...>
{

public:

    NeuralNetwork() = delete;
};

/*
template<size_t size1, size_t size2>
class NeuralNetwork<size1, size2> : public NeuralNetworkBase<size1, size2>
{
using base =  NeuralNetworkBase<size1, size2>;
//region constexpr static

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

    NeuralNetwork() : base()
    {
    }

//endregion

//region methods

    DeltaLayer<size1, size2>
    compute_layer_changes(Vector<double, size2> input,
                          Vector<double, size1> prediction,
                          Vector<double, size1> expected_result)
    {
        DeltaLayer<size1, size2> d_layer;
        return d_layer;
    }

//endregion

};
*/
template<template<size_t...> typename NeuralNetworkInsideT,
        size_t size1, size_t size2, size_t... sizes>
class NeuralNetwork<NeuralNetworkInsideT, size1, size2, sizes...>
        : public NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>
{

//region constexpr static
public:
    using base =  NeuralNetworkBase<NeuralNetworkInsideT, size1, size2, sizes...>;
    using this_t = NeuralNetwork<NeuralNetworkInsideT, size1, size2, sizes...>;
    using other_layers_t = NeuralNetworkInsideT<size2, sizes...>;

    constexpr static size_t last_size()
    {
        return base::last_size();
    }

    using input_t = typename base::input_t;
    using output_t = typename base::output_t;


//endregion

//region constructors

public:
    NeuralNetwork(DMatrix<size2, size1> weights_, DVector<size1> biases_, other_layers_t other_layers_) :
            NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>(weights_, biases_, other_layers_)
    {}

    NeuralNetwork() : NeuralNetworkBase<NeuralNetworkInside, size1, size2, sizes...>()
    {}

//endregion

//region methods

    Vector<double, size1> predict(const Vector<double, last_size()> &input) const
    {
        Vector<double, size2> semi_result = this->other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return softmax(this->weights * semi_result + this->biases);
    }

    RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...>
    feedforward(const input_t &input) const
    {
        RecTuple<Vector<double, size2>, Vector<double, sizes>...> semi_result =
                this->other_layers.inner_feedforward(input);

        return {(softmax(this->weights * (semi_result.head) + this->biases)), semi_result};
    }

    double test_one_input(const input_t &input, const output_t &expected_output)
    {
        return base::right_or_wrong_loss(predict(input),
                                         expected_output);
    }

    double test(const vector<input_t> &inputs, const vector<output_t> &expected_outputs) const
    {
        assert(inputs.size() == expected_outputs.size());
        double acc = 0;
        for (size_t x = 0; x < inputs.size(); x++)
        {
            auto prediction = predict(inputs[x]);
            //cout << DMatrix<2, 10>({prediction, expected_output[x]}) << endl;
            //cout << get<1>(prediction.max_index()) << " ";
            //cout << get<1>(expected_output[x].max_index()) << endl;
            //cout << prediction << endl;
            //cout << expected_output[x] << endl << endl;
            acc += test_one_input(inputs[x],
                                  expected_outputs[x]); //base::sq_err_loss(predict(inputs[x]), expected_output[x]);
        }
        int a = inputs.size();
        return acc / inputs.size();
    }

//endregion

};

#endif //NEURAL_NETWORK_NEURALNETWORK_H

