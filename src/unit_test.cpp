//
// Created by emile on 09/06/19.
//

#include <iostream>
#include <cassert>
#include "NeuralNetwork.h"
//#include "Matrixtibo.h"

using namespace std;

void test_matrix()
{
    Vector<int, 2> a(1, 2);
    RowVector<int, 2> a_(1, 2);
    Vector<int, 1> a__(5);
    assert(a_ * a == a__);
    Matrix<int, 2, 2> b(1, 2,
                        -1, 1);
    Vector<int, 2> b_(4, 7);
    Vector<int, 2> b__(-3, 15);
    assert(b * b_ == b__);
    //Matrix<float, 5, 3> c(2, 3);
    Vector<float, 3> c_(2, 7, 5);
    cout << "Multiplication tests passed" << endl;

    Matrix<int, 2, 2> b___(2, 3, 0, 2);

    assert(b.fmap(function([](int x)
                           { return x + 1; })) == b___);
    //cout << b.to_string() << endl;
    //cout << (b * a).to_string() << endl;
    std::random_device rd = std::random_device();
    mt19937 e2(rd());

    Matrix<double, 5, 5> randm = Matrix<double, 5, 5>::random(e2, -1, 1);

    cout << "Matrix tests passed." << endl;
}

void test_RecTuple()
{
    RecTuple<int, int> tuple0 = RecTuple<int, int>();
    set_nth<0>(tuple0, 2);
    set_nth<1>(tuple0, 3);
    assert(get_nth<0>(tuple0) == 2);
    assert(get_nth<1>(tuple0) == 3);

    auto tuple1 = RecTuple<Vector<double, 4>, Vector<double, 2>, Vector<double, 3>>();
    set_nth<0>(tuple1, Vector<double, 4>(1, 2, 3, 4));
    set_nth<1>(tuple1, Vector<double, 2>(3, 4));
    set_nth<2>(tuple1, Vector<double, 3>(1, 2, 3));

    auto g0 = get_nth<0>(tuple1);
    assert(g0 == (Vector<double, 4>(1, 2, 3, 4)));
    assert(get_nth<1>(tuple1) == (Vector<double, 2>(3, 4)));
    assert(get_nth<2>(tuple1) == (Vector<double, 3>(1, 2, 3)));

    cout << "Tuple tests passed" << endl;

}

/*= default;
void test_MatrixTibo()
{
    double v[] = {1, 1, 1, 1};
    Matrix A = Matrix(2, 2);
    A.setValues(v);
    Matrix B = Matrix(2, 2);
    B.setValues(v);
    Matrix C = A + B;
    Matrix D = A * 2;
    assert(A + B == A * 2);
}
*/
int main()
{
    test_matrix();
    test_RecTuple();
    //test_MatrixTibo();
    cout << "All tests passed." << endl;
    return 0;
}