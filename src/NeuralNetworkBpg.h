//
// Created by emile on 08/11/2019.
//

#ifndef NEURAL_NETWORK_NEURALNETWORKBPG_H
#define NEURAL_NETWORK_NEURALNETWORKBPG_H

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
#include "NeuralNetwork.h"
#include "NeuralNetworkInsideBpg.h"

using namespace std;


template<size_t... sizes>
class NeuralNetworkBPG final : public NeuralNetwork<NeuralNetworkInsideBPG, sizes...>
{

};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkBPG<size1, size2, sizes...> : public NeuralNetwork<NeuralNetworkInsideBPG, size1, size2, sizes...>
{
public:
    using this_t = NeuralNetworkBPG<size1, size2, sizes...>;
    using base = NeuralNetwork<NeuralNetworkInsideBPG, size1, size2, sizes...>;
    using other_layers_t = NeuralNetworkInsideBPG<size2, sizes...>;
    using input_t = typename base::input_t;
    using output_t = typename base::output_t;

    NeuralNetworkBPG() = default;

private:
    DeltaLayer<size1, size2>
    compute_layer_changes(const Vector<double, size2> &input,
                          const Vector<double, size1> &prediction,
                          const Vector<double, size1> &expected_result,
                          size_t chunk_size) const
    {

        Vector<double, size1> a_delta = (prediction - expected_result) / chunk_size;
        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_input = this->weights.transpose() * a_delta;
        d_layer.d_biases = a_delta;

        return d_layer;
    }

    DeltaNetwork<size1, size2, sizes...>
    backpropagate_one_input(const RecTuple<DVector<size1>, DVector<size2>, DVector<sizes>...> predictions,
                            const DVector<size1> &expected_output,
                            size_t chunk_size) const
    {

        DeltaLayer<size1, size2> d_layer = compute_layer_changes((predictions.tail).head,
                                                                 predictions.head,
                                                                 expected_output,
                                                                 chunk_size);

        DeltaNetwork<size2, sizes...> other_changes =
                this->other_layers.inner_backpropagate_one_input(predictions.tail,
                                                                 predictions.tail.head + d_layer.d_input,
                                                                 chunk_size);

        return DeltaNetwork<size1, size2, sizes...>(d_layer, other_changes);
    }

    tuple<DeltaNetwork<size1, size2, sizes...>, double>
    learn_one_input(const DVector<base::last_size()> &input, const DVector<size1> &expected_output,
                    size_t chunk_size) const
    {
        RecTuple<DVector<size1>, DVector<size2>, DVector<sizes>...> predictions = feedforward(input);
        //cout << predictions.head.to_string()  << endl;
        return {backpropagate_one_input(predictions, expected_output, chunk_size),
                base::right_or_wrong_loss(predictions.head, expected_output)};;
    }

    void learn_one_chunk(typename vector<DVector<base::last_size()>>::const_iterator inputs_begin,
                         typename vector<DVector<base::last_size()>>::const_iterator inputs_end,
                         typename vector<DVector<size1>>::const_iterator expected_output_begin,
                         typename vector<DVector<size1>>::const_iterator expected_output_end)
    {
        size_t chunk_size = inputs_end - inputs_begin;
        assert(chunk_size == expected_output_end - expected_output_begin);
        DeltaNetwork<size1, size2, sizes...> d_accumulator;
        double error_accumulator = 0;
        auto expected_output = expected_output_begin;
        for (auto input = inputs_begin;
             input < inputs_end;
             input += base::max_number_of_cores, expected_output += base::max_number_of_cores)
        {
            array<future<tuple<DeltaNetwork<size1, size2, sizes...>, double>>, base::max_number_of_cores> future_learns;
            for (size_t core = 0; core < base::max_number_of_cores && (input + core) < inputs_end; core++)
            {


                const auto async_callback = bind(&this_t::learn_one_input,
                                                 this,
                                                 ref(*(input + core)),
                                                 ref(*(expected_output + core)),
                                                 chunk_size);
                future_learns.at(core) = async(async_callback);
            }
            for (size_t core = 0; core < base::max_number_of_cores && (input + core) < inputs_end; core++)
            {
                auto r = future_learns.at(core).get();
                d_accumulator += get<0>(r);
                error_accumulator += get<1>(r);
            }

            /*
            auto learn = learn_one_input(*input, *expected_output);
            d_accumulator += get<0>(learn);
            error_accumulator += get<1>(learn);
            */
        }
        cout << error_accumulator / (inputs_end - inputs_begin) << endl;
        //for_each(std::execution::par_unseq, )
        d_accumulator *= base::learning_rate;// / (inputs_end - inputs_begin);
        *this -= d_accumulator;
    }

public:
    void learn(const vector<DVector<base::last_size()>> &inputs,
               const vector<DVector<size1>> &expected_output,
               size_t chunk_size,
               size_t epochs)
    {
        assert(inputs.size() == expected_output.size());
        assert(chunk_size <= inputs.size());
        assert(chunk_size <= epochs);
        auto inputs_b1 = inputs.begin();
        auto inputs_b2 = inputs_b1 + chunk_size;

        auto output_b1 = expected_output.begin();
        auto output_b2 = output_b1 + chunk_size;
        for (size_t e = 0; e < epochs && inputs_b2 <= inputs.end();)
        {
            for (; inputs_b2 <= inputs.end(); inputs_b1 += chunk_size,
                                              inputs_b2 += chunk_size,
                                              output_b1 += chunk_size,
                                              output_b2 += chunk_size,
                    e++)
            {
                //if (inputs_b1 - in)
                //cout << this->to_string() << endl;

                learn_one_chunk(inputs_b1, inputs_b2, output_b1, output_b2);
            }
        }
    }
};

#endif //NEURAL_NETWORK_NEURALNETWORKBPG_H
