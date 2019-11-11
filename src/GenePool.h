//
// Created by emile on 09/11/2019.
//

#ifndef NEURAL_NETWORK_GENEPOOL_H
#define NEURAL_NETWORK_GENEPOOL_H

#include <cstddef>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;

template<size_t size1, size_t size2, size_t... sizes>
class NeuralNetworkGene : public NeuralNetwork<NeuralNetworkInside, size1, size2, sizes...>
{
public:
    double eval;
    //region constexpr static
public:
    using base =  NeuralNetwork<NeuralNetworkInside, size1, size2, sizes...>;
    using this_t = NeuralNetworkGene<size1, size2, sizes...>;
    using other_layers_t = typename base::other_layers_t;

    constexpr static size_t last_size()
    {
        return base::last_size();
    }

    using input_t = typename base::input_t;
    using output_t = typename base::output_t;

//endregion

//region constructors

public:
    NeuralNetworkGene(DMatrix<size2, size1> weights_, DVector<size1> biases_, other_layers_t other_layers_) :
            base(weights_, biases_, other_layers_), eval(0)
    {}

    NeuralNetworkGene() : base(), eval(0)
    {}

//endregion

//region operators

    bool operator<(const this_t &other) const
    {
        return eval < other.eval;
    }

    /*this_t &operator+=(const NeuralNetworkBasic<size1, size2, sizes...> &other)
    {
        this->weights += other.weights;
        this->biases += other.biases;
        this->other_layers += other.other_layers;
        return *this;
    }*/

//endregion

//region methods

    void mutate(mt19937 &seed, double mutation_rate)
    {
        DeltaNetwork<size1, size2, sizes...> mutation = random_factory<DeltaNetwork, size1, size2, sizes...>(seed, 0,
                                                                                                             mutation_rate,
                                                                                                             0,
                                                                                                             mutation_rate);
        *this += mutation;
    }

};

template<size_t batch_size, size_t... sizes>
class GenePool
{
    using NeuralNetworkT = NeuralNetworkGene<sizes...>;
    using input_t = typename NeuralNetworkT::input_t;
    using output_t = typename NeuralNetworkT::output_t;

private:
    array<NeuralNetworkGene<sizes...>, batch_size> pool;
    random_device rd;
    mt19937 seed;

    void
    evaluate_helper(const vector<input_t> &inputs, const vector<output_t> &expected_outputs, size_t begin, size_t end,
                    size_t number_of_input_to_try)
    {
        uniform_int_distribution<size_t> dist(0, inputs.size() - 1);
        for (size_t i = begin; i < end; i++)
        {
            double v = 0;
            for (size_t j = 0; j < number_of_input_to_try; j++)
            {
                size_t k = dist(seed);
                const auto &input = inputs[k];
                const auto &expected_output = expected_outputs[k];
                v += pool[i].test_one_input(input, expected_output);
            }
            double v_ = v / number_of_input_to_try;
            pool[i].eval = v_;
        }
    }

public:
    GenePool() : rd(), seed(rd())
    {
        for (auto &i : pool)
        {
            i = random_factory<NeuralNetworkGene, sizes...>(seed, 0, 1);
        }
    }

    void evaluate_pool(const vector<input_t> &inputs, const vector<output_t> &expected_outputs,
                       size_t number_of_input_to_try, size_t n_cores = 16)
    {
        vector<future<void>> threads;

        for (size_t i = 0; i < n_cores; i++)
        {
            size_t begin = (i * batch_size) / n_cores;
            size_t end = ((i + 1) * batch_size) / n_cores;
            const auto async_callback = bind(&GenePool<batch_size, sizes...>::evaluate_helper,
                                             this,
                                             inputs,
                                             expected_outputs,
                                             begin, end, number_of_input_to_try);
            threads.push_back(async(async_callback));
        }
        for (auto &i : threads)
        {
            i.get();
        }
    }

    array<double, batch_size> get_evaluation()
    {
        array<double, batch_size> r;
        for (size_t i = 0; i < batch_size; i++) r[i] = pool[i].eval;
        return r;
    }

    void sort_pool()
    {
        sort(pool.begin(), pool.end());
    }

    void set_to_next_generation(const vector<input_t> &inputs, const vector<output_t> expected_outputs,
                                double pressure, double mutation_rate, size_t number_of_input_to_try,
                                size_t n_cores = 16)
    {
        evaluate_pool(inputs, expected_outputs, number_of_input_to_try, n_cores);
        sort_pool();
        size_t number_of_survivors = batch_size * pressure;
        uniform_int_distribution<size_t> random_survivor(0, number_of_survivors - 1);
        for (size_t i = number_of_survivors; i < batch_size; i++)
        {
            pool[i] = pool[random_survivor(seed)];
            pool[i].mutate(seed, mutation_rate);
        }
    }


};


#endif //NEURAL_NETWORK_GENEPOOL_H

