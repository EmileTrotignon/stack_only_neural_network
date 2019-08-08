//
// Created by emile on 02/08/2019.
//

#ifndef NEURAL_NETWORK_NEURALNETWORKBASE_H
#define NEURAL_NETWORK_NEURALNETWORKBASE_H


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
#include "DNetwork.h"

using namespace std;

constexpr static double low_bound = -1;
constexpr static double up_bound = 1;

template<size_t ...sizes>
class NeuralNetworkInside;

//region NeuralNetworkBase
template<size_t ...sizes>
class NeuralNetworkBase final
{

public:

    NeuralNetworkBase() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetworkBase<size1, size2>
{
public:

    Matrix<double, size2, size1> weights;
    Vector<double, size1> biases;

public:

    NeuralNetworkBase()
    {

        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        weights.iter([&](double &x)
                     {
                         x = dist(e2);
                     });
        biases.iter([&](double &x)
                    {
                        x = dist(e2);
                    });
    }

    NeuralNetworkBase<size1, size2> &operator+=(D_Network<size1, size2> d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        return *this;
    }

    string to_string()
    {
        return weights.to_string();
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
class NeuralNetworkBase<size1, size2, sizes...>
{
protected:
    Matrix<double, size2, size1> weights;
    Vector<double, size1> biases;
    NeuralNetworkInside<size2, sizes...> other_layers;
    static constexpr double learning_rate = 0.5;


public:

    NeuralNetworkBase() : other_layers(), biases({0})
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        weights.iter([&](double &x)
                     {

                         x = dist(e2);
                     });
    }

    string to_string()
    {
        return weights.to_string() + "\n" + other_layers.to_string();
    }

    NeuralNetworkBase<size1, size2, sizes...> &operator+=(D_Network<size1, size2, sizes...> d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        other_layers += d_network.other_layers;

        return *this;
    }

    friend ostream &operator<<(ostream &s, const NeuralNetworkBase<size1, size2, sizes...> &n)
    {
        s << n.to_string();
        return s;
    }

    constexpr static size_t get_number_of_layers()
    {
        return 2 + sizeof...(sizes); //1 + NeuralNetworkBase<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if (i == 0)
        {
            return size1;
        } else
        {
            return NeuralNetworkBase<size2, sizes...>::template NeuralNetworkBase<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return NeuralNetworkBase<size2, sizes...>::last_size();
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

    static double sq_err_loss(Vector<double, size1> prediction, Vector<double, size1> real_value)
    {
        return (real_value - prediction).fmap(function([](double x)
                                                       { return x * x; })).sum();
    }

    static double cross_entropy_loss(Vector<double, size1> prediction, Vector<double, size1> real_value)
    {
        return -(prediction * (real_value).fmap(log)).sum();
    }
};

//endregion



#endif //NEURAL_NETWORK_NEURALNETWORKBASE_H
