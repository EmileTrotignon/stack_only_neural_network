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
#include "Matrix.h"
#include "relevant_math.h"


using namespace std;

constexpr static double low_bound = -1;
constexpr static double up_bound = 1;


template<size_t size1, size_t size2>
class D_Layer
{
public:
    Matrix<double, size2, size1> d_weights;
    Matrix<double, 1, size2> d_input;
    Matrix<double, 1, size2> d_biases;

    D_Layer(Matrix<double, size2, size1> d_weights_, Matrix<double, 1, size2> d_input_,
            Matrix<double, 1, size2> d_biases_) : d_weights(d_weights_), d_biases(d_biases_), d_input(d_input_)
    {

    }

};


template<size_t... sizes>
class D_Network
{

public:

    D_Network() = delete;
};

template<size_t size1, size_t size2>
class D_Network<size1, size2>
{
public:
    D_Layer<size1, size2> d_layer;

    explicit D_Network(D_Layer<size1, size2> d_layer_) : d_layer(d_layer_)
    {

    }

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }
};

template<size_t size1, size_t size2, size_t... sizes>
class D_Network<size1, size2, sizes...>
{
public:
    D_Layer<size1, size2> d_layer;
    D_Network<size1, sizes...> other_layers;

    explicit D_Network(D_Layer<size1, size2> d_layer_, D_Network<size2, sizes...> other_layers_) : d_layer(d_layer_),
                                                                                                   other_layers(
                                                                                                           other_layers_)
    {

    }

    constexpr static size_t get_number_of_layers()
    {
        return 1 + D_Layer<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if (i == 0)
        {
            return size1;
        } else
        {
            return D_Network<size2, sizes...>::template get_size<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return D_Network<size2, sizes...>::last_size();
    }

};

template<size_t ...sizes>
class NeuralNetwork
{

public:

    NeuralNetwork() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetwork<size1, size2>
{
private:

    Matrix<double, size2, size1> first_weights;
    Matrix<double, 1, size1> first_biases;

public:

    NeuralNetwork()
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        first_weights.iter([&](double &x)
                           {
                               x = dist(e2);
                           });
        first_biases.iter([&](double &x)
                          {
                              x = dist(e2);
                          });
    }

    string to_string()
    {
        return first_weights.to_string();
    }

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }

    Matrix<double, 1, size1> inner_predict(Matrix<double, 1, size2> input)
    {
        return (first_weights * input + first_biases).fmap(function(sigmoid));
    }


    D_Layer<size1, size2>
    compute_layer_changes(Matrix<double, 1, size2> input, Matrix<double, 1, size1> prediction,
                          Matrix<double, 1, size1> expected_result)
    {
        Matrix<double, size2, size1> each_column_is_input({input});
        for (size_t x = 0; x < size2; x++)
        {
            each_column_is_input.column(x) = input;
        }

        D_Layer<size1, size2> d_layer;
        Matrix<double, size2, size1> ones({1});
        d_layer.d_weights =
                2 * (prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * each_column_is_input;

        d_layer.d_input =
                2 *
                ((prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * first_weights).column_sum();
        Matrix<double, 1, size2> d_biases =
                2 * ((prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * ones).column_sum();

        return d_layer;


    }


    D_Network<size1, size2>
    inner_backpropagate_one_input(tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>>
                                  predictions, Matrix<double, 1, size1> expected_output)
    {
        D_Layer d_layer = compute_layer_changes(get<1>(predictions), get<0>(predictions), expected_output);
        return D_Network<size1, size2>(d_layer);

    }
};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetwork<size1, size2, sizes...>
{

    Matrix<double, size2, size1> weights;
    Matrix<double, 1, size1> first_biases;
    NeuralNetwork<size2, sizes...> other_layers;
    static constexpr double learning_rate = 0.5;


public:

    NeuralNetwork() : other_layers()
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        weights.iter([&](double &x)
                     {

                         x = dist(e2);
                     });
        first_biases.iter([&](double &x)
                          {
                              x = dist(e2);
                          });
    }

    string to_string()
    {
        return weights.to_string() + "\n" + other_layers.to_string();
    }

    friend ostream &operator<<(ostream &s, NeuralNetwork<size1, size2, sizes...> n)
    {
        s << n.to_string();
        return s;
    }

    constexpr static size_t get_number_of_layers()
    {
        return 1 + NeuralNetwork<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if (i == 0)
        {
            return size1;
        } else
        {
            return NeuralNetwork<size2, sizes...>::template get_size<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return NeuralNetwork<size2, sizes...>::last_size();
    }

    Matrix<double, 1, size1> predict(Matrix<double, 1, last_size()> input)
    {
        Matrix<double, 1, size2> semi_result = other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return softmax(weights * semi_result + first_biases);
    }


    tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...>
    feedforward(Matrix<double, 1, last_size()> input)
    {
        tuple<Matrix<double, 1, size2>, Matrix<double, 1, sizes>...> semi_result = other_layers.inner_feedforward(
                input);
        return tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...>(
                softmax(get<0>(semi_result) * weights + first_biases), semi_result);
    }

    template<size_t i>
    Matrix<double, get_size<i + 1>(), get_size<i>()> get_matrix()
    {
        if (i == 0)
        {
            return weights;
        } else
        {
            return other_layers.template get_size<i - 1>();
        }
    }

    static double sq_err_loss(Matrix<double, 1, size1> prediction, Matrix<double, 1, size1> real_value)
    {
        return (prediction - real_value).map(function([](double x)
                                                      { return x * x; })).sum();
    }


    static double cross_entropy_loss(Matrix<double, 1, size1> prediction, Matrix<double, 1, size1> real_value)
    {
        return -(prediction * (real_value).fmap(log)).sum();
    }
//protected:

    Matrix<double, 1, size1> apply_one_iter(Matrix<double, 1, size2> m)
    {
        return (weights * m + first_biases).fmap(function(sigmoid));
    }

    Matrix<double, 1, size1> inner_predict(Matrix<double, 1, last_size()> input)
    {
        Matrix<double, 1, size2> semi_result = other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return apply_one_iter(semi_result);
    }

    tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...>
    inner_feedforward(Matrix<double, 1, last_size()> input)
    {
        tuple<Matrix<double, 1, size2>, Matrix<double, 1, sizes>...> semi_result = other_layers.inner_feedforward(
                input);
        return tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...>(
                apply_one_iter(get<0>(semi_result)), semi_result);
    }

    D_Layer<size1, size2>
    compute_layer_changes(Matrix<double, 1, size2> input, Matrix<double, 1, size1> prediction,
                          Matrix<double, 1, size1> expected_result)
    {
        Matrix<double, size2, size1> each_column_is_input({input});
        for (size_t x = 0; x < size2; x++)
        {
            each_column_is_input.column(x) = input;
        }

        D_Layer<size1, size2> d_layer;
        Matrix<double, size2, size1> ones({1});
        d_layer.d_weights =
                2 * (prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * each_column_is_input;

        d_layer.d_input =
                2 * ((prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * weights).column_sum();
        Matrix<double, 1, size2> d_biases =
                2 * ((prediction - expected_result) * prediction.fmap(function(sigmoid_derv)) * ones).column_sum();

        return d_layer;


    }

    D_Network<size1, size2, sizes...>
    inner_backpropagate_one_input(tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...>
                                  predictions, Matrix<double, 1, size1> expected_output)
    {
        D_Layer d_layer = compute_layer_changes(get<1>(predictions), get<0>(predictions), expected_output);

        tuple<Matrix<double, 1, size1>, Matrix<double, 1, sizes>...> other_predictions;
        for (size_t x = 1; x < get_number_of_layers(); x++)
        {
            get<x - 1>(other_predictions) = get<x>(predictions);
        }
        D_Network<size2, sizes...> other_changes = other_layers.backpropagate_one_input(predictions,
                                                                                        get<1>(predictions) -
                                                                                        d_layer.d_input);

        return D_Network<size1, size2, sizes...>(d_layer, other_changes);

    }

    D_Network<size1, size2, sizes...>
    learn(Matrix<double, 1, last_size()> input, Matrix<double, 1, size2> expected_output)
    {
        tuple<Matrix<double, 1, size1>, Matrix<double, 1, size2>, Matrix<double, 1, sizes>...> predictions = feedforward(
                input);
        return inner_backpropagate_one_input(predictions, expected_output);
    }
};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
