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
#include <filesystem>
#include <fstream>
#include "Matrix.h"
#include "relevant_math.h"
#include "RecTuple.h"
#include "DeltaNetwork.h"

using namespace std;

constexpr static double low_bound = -1;
constexpr static double up_bound = 1;

template<size_t W, size_t H> using DMatrix = Matrix<double, W, H>;
template<size_t H> using DVector = Vector<double, H>;


template<size_t ...sizes>
class NeuralNetworkInside;

template<size_t ...sizes>
class NeuralNetworkBase final
{
public:
    NeuralNetworkBase() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetworkBase<size1, size2>
{

//region members

protected:
    DMatrix<size2, size1> weights;
    DVector<size1> biases;

//endregion

//region constructors

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

//endregion

//region operators

public:
    NeuralNetworkBase<size1, size2> &operator+=(DeltaNetwork<size1, size2> d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        return *this;
    }

//endregion

//region methods

public:
    string to_string()
    {
        return weights.to_string();
    }

//endregion

//region constexpr static

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

};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkBase<size1, size2, sizes...>
{

//region constexpr static

protected :
    static constexpr double learning_rate = 0.1;

public:
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

//endregion

//region members

public:
    constexpr static size_t max_number_of_cores = 16;

protected:
    DMatrix<size2, size1> weights;
    DVector<size1> biases;
    NeuralNetworkInside<size2, sizes...> other_layers;

//endregion

//region constructors

public:
    NeuralNetworkBase() : other_layers(), biases(MatrixFactory::uniform<double, 1, size1>(0))
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        weights.iter([&](double &x)
                     {
                         x = dist(e2);
                     });
    }

//endregion

//region factories

public:
    static NeuralNetworkBase<size1, size2, sizes...> deserialize(const filesystem::path &path)
    {
        ifstream file;
        file.open(path, ios_base::binary);
        NeuralNetworkBase<size1, size2, sizes...> r;
        file >> r;
        return r;
    }

//endregion

//region operators

public:
    NeuralNetworkBase<size1, size2, sizes...> &operator+=(const DeltaNetwork<size1, size2, sizes...> &d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        other_layers += d_network.other_layers;

        return *this;
    }

    /*friend ostream &operator<<(ostream &s, const NeuralNetworkBase<size1, size2, sizes...> &n)
    {
        s << n.to_string();
        return s;
    }*/

//endregion


//region methods

    [[nodiscard]] string to_string() const
    {
        return weights.to_string() + "\n" + other_layers.to_string();
    }

    template<size_t i>
    Matrix<double, get_size<i + 1>(), get_size<i>()> get_weight() const
    {
        if (i == 0)
        {
            return weights;
        } else
        {
            return other_layers.template get_size<i - 1>();
        }
    }

    void serialize(const filesystem::path &path)
    {
        ofstream file;
        file.open(path, ios_base::binary);
        file << *this;
    }

//endregion

//region static

    static double sq_err_loss(DVector<size1> prediction, DVector<size1> real_value)
    {
        return (real_value - prediction).fmap(function([](double x)
                                                       { return x * x; })).sum();
    }

    static double cross_entropy_loss(DVector<size1> prediction, DVector<size1> real_value)
    {
        return -(prediction * (real_value).fmap(log)).sum();
    }

    static double right_or_wrong_loss(DVector<size1> prediction, DVector<size1> real_value)
    {
        return real_value.max_index() == prediction.max_index() ? 1 : 0;
    }

//endregion

};

#endif //NEURAL_NETWORK_NEURALNETWORKBASE_H
