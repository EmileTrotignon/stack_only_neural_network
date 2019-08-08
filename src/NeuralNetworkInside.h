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
#include "DNetwork.h"
#include "NeuralNetworkBase.h"

//using namespace std;

//region NeuralNetworkInside

template<size_t ...sizes>
class NeuralNetworkInside
{

public:

    NeuralNetworkInside() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetworkInside<size1, size2> : public NeuralNetworkBase<size1, size2>
{
public:

    NeuralNetworkInside()
    {

        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        this->weights.iter([&](double &x)
                           {
                               x = dist(e2);
                           });
        this->biases.iter([&](double &x)
                          {
                              x = dist(e2);
                          });
    }

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }

    Vector<double, size1> inner_predict(Vector<double, size2> input)
    {
        return (this->weights * input + this->biases).fmap(function(sigmoid));
    }

    RecTuple<Vector<double, size1>, Vector<double, size2>> inner_feedforward(Vector<double, size2> input)
    {
        return {inner_predict(input), {input}};
    }

    D_Layer<size1, size2>
    compute_layer_changes(Vector<double, size2> input, Vector<double, size1> prediction,
                          Vector<double, size1> expected_result)
    {

        Vector<double, size1> a_delta = Vector<double, size1>::element_by_element_product(
                (expected_result - prediction), prediction.fmap(function(sigmoid_derv)));

        D_Layer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

//        d_layer.d_input = this->weights.transpose() * input;
        d_layer.d_biases = a_delta;
        return d_layer;


    }

    D_Network<size1, size2>
    inner_backpropagate_one_input(RecTuple<Vector<double, size1>, Vector<double, size2>> predictions,
                                  Vector<double, size1> expected_output)
    {
        D_Layer d_layer = compute_layer_changes((predictions.tail.head), (predictions.head), expected_output);
        return D_Network<size1, size2>(d_layer);

    }
};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkInside<size1, size2, sizes...> : public NeuralNetworkBase<size1, size2, sizes...>
{

    Matrix<double, size2, size1> weights;
    Vector<double, size1> biases;
    NeuralNetworkInside<size2, sizes...> other_layers;
    static constexpr double learning_rate = 0.5;


public:

    NeuralNetworkInside() : other_layers(), biases({0})
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        weights.iter([&](double &x)
                     {

                         x = dist(e2);
                     });
    }

    friend ostream &operator<<(ostream &s, const NeuralNetworkInside<size1, size2, sizes...> &n)
    {
        s << n.to_string();
        return s;
    }

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

    Vector<double, size1> predict(Vector<double, last_size()> input)
    {
        Vector<double, size2> semi_result = other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return softmax(weights * semi_result + biases);
    }

    Vector<double, size1> apply_one_iter(Vector<double, size2> m)
    {
        return (weights * m + biases).fmap(function(sigmoid));
    }

    RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...>
    inner_feedforward(Vector<double, last_size()> input)
    {
        RecTuple<Vector<double, size2>, Vector<double, sizes>...>
                semi_result = other_layers.inner_feedforward(input);
        auto r = apply_one_iter(semi_result.head);

        //if constexpr (tuple_size<)
        return {r, semi_result};
    }

    //region protected

    Vector<double, size1> inner_predict(Vector<double, last_size()> input)
    {
        Vector<double, size2> semi_result = other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return apply_one_iter(semi_result);
    }


    D_Layer<size1, size2>
    compute_layer_changes(Vector<double, size2> input, Vector<double, size1> prediction,
                          Vector<double, size1> expected_result)
    {
        Vector<double, size1> a_delta = Vector<double, size1>::element_by_element_product(
                (expected_result - prediction), prediction.fmap(function(sigmoid_derv)));

        D_Layer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_input = this->weights.transpose() * input;
        d_layer.d_biases = a_delta;
        return d_layer;


    }

    D_Network<size1, size2, sizes...>
    inner_backpropagate_one_input(RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...>
                                  predictions, Vector<double, size1> expected_output)
    {
        D_Layer d_layer = compute_layer_changes(predictions.tail.head, predictions.head, expected_output);

        D_Network<size2, sizes...> other_changes = other_layers.inner_backpropagate_one_input(predictions.tail,
                                                                                              (predictions.tail.head) -
                                                                                              d_layer.d_input);

        return D_Network<size1, size2, sizes...>(d_layer, other_changes);

    }
};

//endregion

#endif //NEURAL_NETWORK_NEURALNETWORKINSIDE_H
