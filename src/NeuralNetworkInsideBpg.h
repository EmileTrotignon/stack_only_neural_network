//
// Created by emile on 08/11/2019.
//

#ifndef NEURAL_NETWORK_NEURALNETWORKINSIDEBPG_H
#define NEURAL_NETWORK_NEURALNETWORKINSIDEBPG_H

#include "NeuralNetworkInside.h"

template<size_t ...sizes>
class NeuralNetworkInsideBPG final : public NeuralNetworkInside<sizes...>
{

};

template<size_t size1, size_t size2>
class NeuralNetworkInsideBPG<size1, size2> : public NeuralNetworkInside<size1, size2>
{
    NeuralNetworkInsideBPG() = default;

    DeltaLayer<size1, size2>
    compute_layer_changes(const DVector<size2> input,
                          const DVector<size1> prediction,
                          const DVector<size1> expected_result,
                          size_t chunk_size) const
    {

        DVector<size1> a_delta = DVector<size1>::element_by_element_product(
                (prediction - expected_result), prediction.fmap(function(sigmoid_derv)));

        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_biases = a_delta;
        return d_layer;


    }

    DeltaNetwork<size1, size2>
    inner_backpropagate_one_input(const RecTuple<DVector<size1>, DVector<size2>> &predictions,
                                  const DVector<size1> &expected_output,
                                  size_t chunk_size) const
    {
        auto a0 = (predictions.tail.head);
        auto a1 = (predictions.head);
        DeltaLayer d_layer = compute_layer_changes(a0, a1, expected_output, chunk_size);
        return DeltaNetwork<size1, size2>(d_layer);

    }
};

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkInsideBPG<size1, size2, sizes...> : public NeuralNetworkInside<size1, size2, sizes...>
{

    using base = NeuralNetworkInside<size1, size2, sizes...>;
    using input_t = typename base::input_t;
    using output_t = typename base::output_t;

    DeltaLayer<size1, size2>
    compute_layer_changes(const DVector<size2> &input,
                          const DVector<size1> &prediction,
                          const DVector<size1> &expected_result,
                          size_t chunk_size) const
    {
        DVector<size1> a_delta = DVector<size1>::element_by_element_product(
                (prediction - expected_result), prediction.fmap(function(sigmoid_derv)));

        DeltaLayer<size1, size2> d_layer;
        d_layer.d_weights = a_delta * input.transpose();

        d_layer.d_input = this->weights.transpose() * a_delta;
        d_layer.d_biases = a_delta;
        return d_layer;
    }

    DeltaNetwork<size1, size2, sizes...>
    inner_backpropagate_one_input(const RecTuple<DVector<size1>, DVector<size2>, DVector<sizes>...> &predictions,
                                  const DVector<size1> &expected_output,
                                  size_t chunk_size) const
    {
        DeltaLayer d_layer = compute_layer_changes(predictions.tail.head,
                                                   predictions.head,
                                                   expected_output,
                                                   chunk_size);

        DeltaNetwork<size2, sizes...> other_changes =
                this->other_layers.inner_backpropagate_one_input(predictions.tail,
                                                                 (predictions.tail.head) + d_layer.d_input);

        return DeltaNetwork<size1, size2, sizes...>(d_layer, other_changes);
    }

};

#endif //NEURAL_NETWORK_NEURALNETWORKINSIDEBPG_H
