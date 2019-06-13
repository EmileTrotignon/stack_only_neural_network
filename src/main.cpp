#include <iostream>
#include <fenv.h>
#include "NeuralNetwork.h"

using namespace std;

int main()
{
    feenableexcept(FE_INVALID | FE_OVERFLOW);
    Matrix<int, 1, 2> m({1, 2});
    NeuralNetwork<5, 3, 7, 8, 2> network;

    cout << network.to_string() << endl << endl;

    cout << (network.predict(Matrix<double, 1, 2>({1, 2}))) << endl;
    return 0;
}