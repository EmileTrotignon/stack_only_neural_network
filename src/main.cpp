#include <iostream>
#include "../mnist-master/include/mnist/mnist_reader.hpp"
#include "../mnist-master/include/mnist/mnist_utils.hpp"
#include <ctime>
#include <sys/resource.h>

//#define MAKE_STACK_ONLY
#include "Matrix.h"

#include "NeuralNetworkBpg.h"
#include "GenePool.h"
#include <tuple>

using namespace std;

void get_dataset_(vector<Matrix<double, 1, 784>> &inputs, vector<Matrix<double, 1, 10>> &outputs,
                  vector<Matrix<double, 1, 784>> &test_inputs, vector<Matrix<double, 1, 10>> &test_outputs)
{
    RecTuple<int> t = RecTuple<int>();
    t.head = 1;
    auto a = t.head;
    auto raw_data = mnist::read_dataset(100000, 1000);

    mnist::normalize_dataset(raw_data);


    for (const auto &raw_image:raw_data.training_images)
    {
        DVector<784> image(Matrix<double, 1, 784>::uniform(0));
        for (size_t x = 0; x < raw_image.size(); x++)
        {
            image.at(0, x) = double(raw_image[x]) / 256;
        }
        inputs.push_back(move(image));
    }

    for (const auto &raw_label:raw_data.training_labels)
    {
        DVector<10> label(DVector<10>::uniform(0));
        label.at(0, raw_label) = 1;
        outputs.push_back(move(label));
    }

    for (const auto &raw_image:raw_data.test_images)
    {
        DVector<784> image(DVector<784>::uniform(0));
        for (size_t x = 0; x < raw_image.size(); x++)
        {
            image.at(0, x) = double(raw_image[x]) / 256;
        }
        test_inputs.push_back(move(image));
    }

    for (const auto &raw_label:raw_data.test_labels)
    {
        DVector<10> label(DVector<10>::uniform(0));
        label.at(0, raw_label) = 1;
        test_outputs.push_back(move(label));
    }
//    return {inputs, outputs, test_inputs, test_outputs};
}

template<size_t H>
DVector<H> get_Vector(istream &in_stream)
{
    DVector<H> r;
    for (size_t i = 0; i < H; i++)
    {
        in_stream >> r.at(0, i);
    }
    return r;
}

void get_dataset(vector<DVector<64>> &inputs, vector<DVector<10>> &outputs,
                 vector<DVector<64>> &test_inputs, vector<DVector<10>> &test_outputs)
{
    ifstream file_inputs("images_train");
    ifstream file_outputs("labels_train");
    ifstream file_test_inputs("images_test");
    ifstream file_test_outputs("labels_test");
    file_inputs.exceptions(std::ifstream::badbit);
    file_outputs.exceptions(std::ifstream::badbit);
    file_test_inputs.exceptions(std::ifstream::badbit);
    file_test_outputs.exceptions(std::ifstream::badbit);

    size_t l;
    DVector<10> v10;
    DVector<64> v64;
    while (!(file_outputs >> v10).eof())
    {
        outputs.push_back(v10);
        l = outputs.size();
    }
    while (!(file_inputs >> v64).eof())
    {
        inputs.push_back(v64);
        l = inputs.size();
    }
    while (!(file_test_inputs >> v64).eof())
    {
        test_inputs.push_back(v64);
    }
    while (!(file_test_outputs >> v10).eof())
    {
        test_outputs.push_back(v10);
        l = inputs.size();
    }

}

int main()
{
    vector<DVector<64>> inputs;
    vector<DVector<10>> outputs;

    vector<DVector<64>> test_inputs;
    vector<DVector<10>> test_outputs;
    get_dataset(inputs, outputs, test_inputs, test_outputs);
    std::random_device rd = std::random_device();
    cout << rd() << endl;
    mt19937 e2(rd());
    //NeuralNetworkBasic<10, 64> network = random_factory<NeuralNetworkBasic, 10, 128, 64>(e2, 0, 1);
    GenePool<100, 10, 10, 64> pool;
    pool.evaluate_pool(inputs, outputs, 500, 16);
    auto eval = pool.get_evaluation();
    for (auto i:eval)
    {
        cout << i << ", ";
    }
    cout << endl;
    pool.sort_pool();

    for (size_t j = 0; j < 10000; j++)
    {
        pool.sort_pool();
        eval = pool.get_evaluation();
        double s = 0;
        for (auto i:eval)
        {
            //    cout << i << ", ";
            s += i;
        }
        cout << j << " : " << s << endl;
        pool.set_to_next_generation(inputs, outputs, 0.1, 0.3, 100, 16);
    }


    /*
    for (size_t j = 0; j < 10; j++)
    {
        pool.sort_pool();
        auto eval = pool.evaluate_pool(inputs, outputs, 10);
        double s = 0;
        for (auto i:eval)
        {
            cout << i << ", ";
            s += i;
        }
        cout << endl << s << endl;
        pool.set_to_next_generation(inputs, outputs, 0.1, 0.1, 20);
    }
    */
    //network.learn(inputs, outputs, 100, 20000);
    //cout << network.test(test_inputs, test_outputs);
    //tuple<vector<Matrix<double, 1, 784>>, vector<Matrix<double, 1, 10>>> dataset = get_dataset();
/*
    vector<DVector<2>> inputs;
    vector<DVector<2>> outputs;
    for (size_t i = 0; i < 10000; i++)
    {
        DVector<2> v(i % 2, (i + 1) % 2);
        inputs.push_back(v);
        outputs.push_back(v);
    }
    auto network = NeuralNetwork<2, 4, 2>();
    cout << network.to_string() << endl;
    network.learn(inputs, outputs, 10, 100);
    */
    return 0;
}