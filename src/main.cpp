#include <iostream>
#include "../mnist-master/include/mnist/mnist_reader.hpp"
#include "../mnist-master/include/mnist/mnist_utils.hpp"
#include <ctime>
#include <sys/resource.h>

#include "NeuralNetwork.h"
#include <tuple>

using namespace std;

void get_dataset(vector<Matrix<double, 1, 784>> &inputs, vector<Matrix<double, 1, 10>> &outputs,
                 vector<Matrix<double, 1, 784>> &test_inputs, vector<Matrix<double, 1, 10>> &test_outputs)
{
    RecTuple<int> t = RecTuple<int>();
    t.head = 1;
    auto a = t.head;
    auto raw_data = mnist::read_dataset(100000, 1000);

    mnist::normalize_dataset(raw_data);


    for (const auto &raw_image:raw_data.training_images)
    {
        Matrix<double, 1, 784> image(MatrixFactory::uniform<double, 1, 784>(0));
        for (size_t x = 0; x < raw_image.size(); x++)
        {
            image.at(0, x) = double(raw_image[x]) / 256;
        }
        inputs.push_back(move(image));
    }

    for (const auto &raw_label:raw_data.training_labels)
    {
        Matrix<double, 1, 10> label(MatrixFactory::uniform<double, 1, 10>(0));
        label.at(0, raw_label) = 1;
        outputs.push_back(move(label));
    }

    for (const auto &raw_image:raw_data.test_images)
    {
        Matrix<double, 1, 784> image(MatrixFactory::uniform<double, 1, 784>(0));
        for (size_t x = 0; x < raw_image.size(); x++)
        {
            image.at(0, x) = double(raw_image[x]) / 256;
        }
        test_inputs.push_back(move(image));
    }

    for (const auto &raw_label:raw_data.test_labels)
    {
        Matrix<double, 1, 10> label(MatrixFactory::uniform<double, 1, 10>(0));
        label.at(0, raw_label) = 1;
        test_outputs.push_back(move(label));
    }
//    return {inputs, outputs, test_inputs, test_outputs};
}

int main()
{
    /*vector<Matrix<double, 1, 784>> inputs;
    vector<Matrix<double, 1, 10>> outputs;

    vector<Matrix<double, 1, 784>> test_inputs;
    vector<Matrix<double, 1, 10>> test_outputs;
    get_dataset(inputs, outputs, test_inputs, test_outputs);
    struct rlimit rlim{};
    getrlimit(RLIMIT_STACK, &rlim);
    cout << rlim.rlim_cur << endl;
    auto network = new NeuralNetwork<10, 100, 1000, 784>();
    network->learn(inputs, outputs, 100, 2000);
    cout << network->test(test_inputs, test_outputs);*/
    //tuple<vector<Matrix<double, 1, 784>>, vector<Matrix<double, 1, 10>>> dataset = get_dataset();

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
    network.learn(inputs, outputs, 10, 1000);
    return 0;
}