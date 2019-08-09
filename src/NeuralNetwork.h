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
#include "Matrix.h"
#include "relevant_math.h"
#include "RecTuple.h"
#include "DeltaNetwork.h"
#include "NeuralNetworkBase.h"
#include "NeuralNetworkInside.h"
using namespace std;

//region NeuralNetwork

template<size_t ...sizes>
class NeuralNetwork : public NeuralNetworkBase<sizes...>
{

public:

    NeuralNetwork() = delete;
};

template<size_t size1, size_t size2>
class NeuralNetwork<size1, size2> : public NeuralNetworkBase<size1, size2>
{
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

    NeuralNetwork()
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        this->first_weights.iter([&](double &x)
                           {
                               x = dist(e2);
                           });
        this->biases.iter([&](double &x)
                          {
                              x = dist(e2);
                          });
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

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetwork<size1, size2, sizes...> : public NeuralNetworkBase<size1, size2, sizes...>
{
//region constexpr static
    using base =  NeuralNetworkBase<size1, size2, sizes...>;

    constexpr static size_t last_size()
    {
        return NeuralNetworkBase<size1, size2, sizes...>::last_size();
    }

//endregion
//region members
    size_t chunk_size;
//endregion
//region constructors
public:
    NeuralNetwork()
    {
        std::random_device rd;
        mt19937 e2(rd());
        normal_distribution<double> dist(low_bound, up_bound);
        this->weights.iter([&](double &x)
                     {
                         x = dist(e2);
                     });
        chunk_size = 0;
    }
//endregion

//region operators

    NeuralNetwork<size1, size2, sizes...> &operator+=(const DeltaNetwork<size1, size2, sizes...> &d_network)
    {
        this->weights += d_network.d_layer.d_weights;
        this->biases += d_network.d_layer.d_biases;

        this->other_layers += d_network.other_layers;

        return *this;
    }

//endregion

//region methods

    Vector<double, size1> predict(Vector<double, last_size()> input) const
    {
        Vector<double, size2> semi_result = this->other_layers.inner_predict(input);
        //cout << semi_result << endl;
        return softmax(this->weights * semi_result + this->biases);
    }

    RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...>
    feedforward(Vector<double, last_size()> input) const
    {
        RecTuple<Vector<double, size2>, Vector<double, sizes>...> semi_result =
                this->other_layers.inner_feedforward(input);

        return {(softmax(this->weights * (semi_result.head) + this->biases)), semi_result};
    }

    DeltaLayer<size1, size2>
    compute_layer_changes(Vector<double, size2> input, Vector<double, size1> prediction,
                          Vector<double, size1> expected_result) const
    {

        Vector<double, size1> a_delta = (expected_result - prediction) / (double) chunk_size;
        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_input = this->weights.transpose() * a_delta;
        d_layer.d_biases = a_delta;

        return d_layer;


    }

    DeltaNetwork<size1, size2, sizes...>
    backpropagate_one_input(
            RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...> predictions,
            Vector<double, size1> expected_output) const
    {
        Vector<double, size1> a0 = (predictions.head);

        Vector<double, size2> a1 = (predictions.tail).head;

        DeltaLayer<size1, size2> d_layer = compute_layer_changes(a1, a0, expected_output);

        DeltaNetwork<size2, sizes...> other_changes = this->other_layers.inner_backpropagate_one_input(predictions.tail,
                                                                                                    predictions.tail.head -
                                                                                        d_layer.d_input);

        return DeltaNetwork<size1, size2, sizes...>(d_layer, other_changes);

    }

    DeltaNetwork<size1, size2, sizes...>
    learn_one_input(Vector<double, last_size()> input, Vector<double, size1> expected_output)
    {
        RecTuple<Vector<double, size1>, Vector<double, size2>, Vector<double, sizes>...> predictions = feedforward(
                input);
        return backpropagate_one_input(predictions, expected_output);
    }

    void learn_one_chunk(typename vector<Vector<double, last_size()>>::iterator inputs_begin,
                         typename vector<Vector<double, last_size()>>::iterator inputs_end,
                         typename vector<Vector<double, size1>>::iterator expected_output_begin,
                         typename vector<Vector<double, size1>>::iterator expected_output_end)
    {
        assert(inputs_end - inputs_begin == expected_output_end - expected_output_begin);

        DeltaNetwork<size1, size2, sizes...> d_accumulator;
        auto expected_output = expected_output_begin;
        for (auto input = inputs_begin; input < inputs_end; input++, expected_output++)
        {
            d_accumulator += learn_one_input(*input, *expected_output);
        }
        d_accumulator *= base::learning_rate / (inputs_end - inputs_begin);
        *this += d_accumulator;
    }

    void
    learn(vector<Vector<double, last_size()>> inputs, vector<Vector<double, size1>> expected_output, size_t chunk_size_,
          size_t epochs)
    {
        chunk_size = chunk_size_;
        assert(inputs.size() == expected_output.size());
        assert(chunk_size <= inputs.size());
        assert(chunk_size <= epochs);
        for (size_t e = 0; e < epochs;)
        {
            auto inputs_b1 = inputs.begin();
            auto output_b1 = expected_output.begin();
            auto output_b2 = output_b1 + chunk_size;
            auto inputs_b2 = inputs_b1 + chunk_size;
            for (; inputs_b2 <= inputs.end(); inputs_b1 += chunk_size,
                                              inputs_b2 += chunk_size,
                                              output_b1 += chunk_size,
                                              output_b2 += chunk_size,
                    e++)
            {
                learn_one_chunk(inputs_b1, inputs_b2, output_b1, output_b2);
            }
        }
    }

    double test(vector<Vector<double, last_size()>> inputs, vector<Vector<double, size1>> expected_output) const
    {
        assert(inputs.size() == expected_output.size());
        double acc = 0;
        for (size_t x = 0; x < inputs.size(); x++)
        {
            auto prediction = predict(inputs[x]);
            cout << Matrix<double, 2, 10>({prediction, expected_output[x]}) << endl;
            cout << get<1>(prediction.max_index()) << " ";
            cout << get<1>(expected_output[x].max_index()) << endl;
            //cout << prediction << endl;
            //cout << expected_output[x] << endl << endl;
            acc += prediction.max_index() == expected_output[x].max_index() ? (cout << "good" << endl), 1
                                                                            : 0; //base::sq_err_loss(predict(inputs[x]), expected_output[x]);
        }
        int a = inputs.size();
        return acc / inputs.size();
    }

//endregion

};

#endif //NEURAL_NETWORK_NEURALNETWORK_H
