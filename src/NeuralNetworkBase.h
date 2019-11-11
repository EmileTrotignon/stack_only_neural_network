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

template<size_t W, size_t H> using DMatrix = Matrix<double, W, H>;
template<size_t H> using DVector = Vector<double, H>;

//template<typename NeuralNetworkT, size_t... sizes>
//NeuralNetworkT random_factory(const mt19937 &e2);

template<template<size_t...> typename NeuralNetworkInsideT, size_t ...sizes>
class NeuralNetworkBase final
{
};

template<template<size_t...> typename NeuralNetworkInsideT, size_t size1, size_t size2>
class NeuralNetworkBase<NeuralNetworkInsideT, size1, size2>
{

//region constexpr static

public:

    using this_t = NeuralNetworkBase<NeuralNetworkInsideT, size1, size2>;
    using weights_t = DMatrix<size2, size1>;
    using biases_t = DVector<size1>;

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }

    using input_t = DVector<last_size()>;
    using output_t = DVector<size1>;

//endregion

//region members

public:
    DMatrix<size2, size1> weights;
    DVector<size1> biases;

//endregion

//region constructors
public:
    NeuralNetworkBase(DMatrix<size2, size1> weights_, DVector<size1> biases_) : weights(weights_), biases(biases_)
    {
    }

    NeuralNetworkBase() : weights(), biases()
    {};
//endregion

//region operators

public:
    this_t &operator+=(DeltaNetwork<size1, size2> d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        return *this;
    }

    this_t &operator-=(DeltaNetwork<size1, size2> d_network)
    {
        weights -= d_network.d_layer.d_weights;
        biases -= d_network.d_layer.d_biases;
        return *this;
    }

//endregion

//region methods

public:
    [[nodiscard]] string to_string() const
    {
        return "Layer :\n\nWeights :\n" + weights.to_string() + "\nBiases:\n\n" + biases.to_string();

    }

//endregion

    //friend this_t random_factory<NeuralNetworkBase, size1, size2>(mt19937 &);

};

template<template<size_t...> typename NeuralNetworkInsideT, size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkBase<NeuralNetworkInsideT, size1, size2, sizes...>
{

//region constexpr static

public:
    using this_t = NeuralNetworkBase<NeuralNetworkInsideT, size1, size2, sizes...>;
    using other_layers_t = NeuralNetworkInsideT<size2, sizes...>;
    using weights_t = DMatrix<size2, size1>;
    using biases_t = DVector<size1>;
    static constexpr double learning_rate = 0.5;

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
            return other_layers_t::template NeuralNetworkBase<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return other_layers_t::last_size();
    }

    using input_t = DVector<last_size()>;
    using output_t = DVector<size1>;

//endregion

//region members

public:
    constexpr static size_t max_number_of_cores = 16;

public:
    DMatrix<size2, size1> weights;
    DVector<size1> biases;
    other_layers_t other_layers;

//endregion

//region constructors

public:
    NeuralNetworkBase(DMatrix<size2, size1> weights_, DVector<size1> biases_, other_layers_t other_layers_) :
            weights(weights_), biases(biases_), other_layers(other_layers_)
    {
    }

    NeuralNetworkBase(DMatrix<size2, size1> weights_, DVector<size1> biases_) : weights(weights_), biases(biases_)
    {
        assert(false);
    }

    NeuralNetworkBase() : weights(), biases(), other_layers()
    {};

//endregion

//region factories

public:
    static this_t deserialize(const filesystem::path &path)
    {
        ifstream file;
        file.open(path, ios_base::binary);
        this_t r;
        file >> r;

        int x;
        return r;
    }



//endregion

//region operators

public:
    this_t &operator+=(const DeltaNetwork<size1, size2, sizes...> &d_network)
    {
        weights += d_network.d_layer.d_weights;
        biases += d_network.d_layer.d_biases;
        other_layers += d_network.other_layers;

        return *this;
    }

    this_t &operator-=(const DeltaNetwork<size1, size2, sizes...> &d_network)
    {
        weights -= d_network.d_layer.d_weights;
        biases -= d_network.d_layer.d_biases;
        other_layers -= d_network.other_layers;

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
        return "Layer :\nWeights :\n" + weights.to_string() + "\nBiases:\n" + biases.to_string() +
               other_layers.to_string() + "\n\n";
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

    /*static double cross_entropy_loss(DVector<size1> prediction, DVector<size1> real_value)
    {
        return -((real_value.fmap(function(mylog))) * prediction).sum();
    }*/

    static double right_or_wrong_loss(DVector<size1> prediction, DVector<size1> real_value)
    {
        return real_value.max_index() == prediction.max_index() ? 0 : 1;
    }

//endregion

    //friend this_t random_factory<NeuralNetworkBase, size1, size2, sizes...>(mt19937 &);

};

template<size_t... sizes>
class NeuralNetworkBasic : public NeuralNetworkBase<NeuralNetworkBasic, sizes...>
{

};

template<size_t size1, size_t size2>
class NeuralNetworkBasic<size1, size2> : public NeuralNetworkBase<NeuralNetworkBasic, size1, size2>
{
    using this_t =  NeuralNetworkBasic<size1, size2>;
public:

    NeuralNetworkBasic(DMatrix<size2, size1> weights_, DVector<size1> biases_) :
            NeuralNetworkBase<NeuralNetworkBasic, size1, size2>(weights_, biases_)
    {}

    NeuralNetworkBasic() : NeuralNetworkBase<NeuralNetworkBasic, size1, size2>()
    {}
    /*static this_t random_weights(const mt19937 &e2)
    {
        this_t r;
        r.weights = DMatrix<size2, size1>::random(e2, mean, variance);
        r.biases = DVector<size1>::uniform(0);
        return r;
    }*/
};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkBasic<size1, size2, sizes...> : public NeuralNetworkBase<NeuralNetworkBasic, size1, size2, sizes...>
{
public:
    using this_t = NeuralNetworkBasic<size1, size2, sizes...>;
    using other_layers_t = NeuralNetworkBasic<size2, sizes...>;
public:

    NeuralNetworkBasic(DMatrix<size2, size1> weights_, DVector<size1> biases_, other_layers_t other_layers_) :
            NeuralNetworkBase<NeuralNetworkBasic, size1, size2, sizes...>(weights_, biases_, other_layers_)
    {}

    NeuralNetworkBasic() : NeuralNetworkBase<NeuralNetworkBasic, size1, size2, sizes...>()
    {}

};

//template<template<size_t... sizes_> typename NeuralNetworkT, size_t... sizes>
//NeuralNetworkT<sizes...> random_factory(const mt19937 &e2);



template<typename NeuralNetworkT, size_t size1, size_t size2>
NeuralNetworkT random_helper(mt19937 &e2, double mean, double variance, double biases_mean, double biases_variance)
{
    auto biases = NeuralNetworkT::biases_t::random(e2, biases_mean, biases_variance);
    auto weights = NeuralNetworkT::weights_t::random(e2, mean, variance);
    return NeuralNetworkT(weights, biases);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "InfiniteRecursion" //yeah boi its not actually infinite

template<typename NeuralNetworkT, size_t size1, size_t size2, size_t size3, size_t... sizes>
NeuralNetworkT random_helper(mt19937 &e2, double mean, double variance, double biases_mean, double biases_variance)
{
    auto biases = NeuralNetworkT::biases_t::random(e2, biases_mean, biases_variance);
    auto weights = NeuralNetworkT::weights_t::random(e2, mean, variance);
    auto other_layers = random_helper<typename NeuralNetworkT::other_layers_t, size2, size3, sizes...>(e2, mean,
                                                                                                       variance,
                                                                                                       biases_mean,
                                                                                                       biases_variance);
    return NeuralNetworkT(weights, biases, other_layers);
}

#pragma clang diagnostic pop

template<template<size_t... sizes> typename NeuralNetworkTemplate, size_t... sizes>
NeuralNetworkTemplate<sizes...> random_factory(mt19937 &e2, double mean, double variance,
                                               double biases_mean = 0, double biases_variance = 0)
{
    return random_helper<NeuralNetworkTemplate<sizes...>, sizes...>(e2, mean, variance, biases_mean, biases_variance);
}

#endif //NEURAL_NETWORK_NEURALNETWORKBASE_H
