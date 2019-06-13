//
// Created by emile on 09/06/19.
//

#include <iostream>
#include <cassert>
#include "NeuralNetwork.h"

using namespace std;

void test_matrix()
{
    Matrix<int, 1, 2> a({1, 2});
    Matrix<int, 2, 1> a_({1, 2});
    Matrix<int, 1, 1> a__({5});
    assert(a_ * a == a__);
    Matrix<int, 2, 2> b({1, 2, -1, 1});
    Matrix<int, 1, 2> b_({4, 7});
    Matrix<int, 1, 2> b__({-3, 15});
    assert(b * b_ == b__);
    Matrix<float, 5, 3> c({2, 3});
    Matrix<float, 1, 3> c_({2, 7, 5});
    cout << "Multiplication tests passed" << endl;

    Matrix<int, 2, 2> b___({2, 3, 0, 2});

    assert(b.fmap(function([](int x)
                           { return x + 1; })) == b___);
    //cout << b.to_string() << endl;
    //cout << (b * a).to_string() << endl;
    cout << "Matrix tests passed." << endl;
}

int main()
{
    test_matrix();
    cout << "All tests passed." << endl;
    return 0;
}