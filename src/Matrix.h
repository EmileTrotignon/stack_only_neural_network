//
// Created by emile on 07/06/19.
//

#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <iostream>
#include <array>
#include <functional>

using namespace std;


template<class T, size_t W, size_t H>
class Matrix;

template<class T, size_t H> using Vector = Matrix<T, 1, H>;

template<class T, size_t W> using RowVector = Matrix<T, W, 1>;


template<class T, size_t W, size_t H>
class Matrix
{
private:
    array<array<T, H>, W> arr;
public:
    Matrix() = default;

    explicit Matrix(array<array<T, H>, W> arr_) : arr(arr_)
    {

    }

    Matrix(const Matrix<T, W, H> &other)
    {
        arr = array<array<T, H>, W>();
        std::copy(other.arr.begin(), other.arr.end(), arr.begin());
    }

    explicit Matrix(Matrix<T, H, W> &&other)
    {
        arr = other.arr();
        other.arr = nullptr;
    }

    Matrix<T, W, H> &operator=(const Matrix<T, W, H> &other) noexcept
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                at(x, y) = other.at(x, y);
            }
        }
        return *this;
    }

    Matrix<T, W, H> &operator=(Matrix<T, W, H> &&other) noexcept
    {
        arr = other.arr;
        other.arr = array<array<T, H>, W>();;
        return *this;
    }

    explicit Matrix(array<Matrix<T, 1, H>, W> am)
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                arr.at(x, y) = am.at(x).at(1, y);
            }
        }
    }

    T at(size_t x, size_t y) const
    {
        return arr.at(x).at(y);
    }

    T &at(size_t x, size_t y)
    {
        return arr.at(x).at(y);
    }

    array<T, H> column(size_t x) const
    {
        return arr.at(x);
    }


    array<T, H> &column(size_t x)
    {
        return arr.at(x);
    }

    array<T, W> line(size_t y) const
    {
        array<T, W> r;
        for (size_t x = 0; x < W; x++)
        {
            r.at(x) = at(x, y);
        }
        return r;
    }

    Matrix<T, H, W> transpose()
    {
        Matrix<T, H, W> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(y, x) = at(x, y);
            }
        }
        return r;
    }


    bool operator==(Matrix<T, W, H> other) const
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                if (at(x, y) != other.at(x, y)) return false;
            }
        }
        return true;
    }

    Matrix<T, W, H> operator+(const Matrix<T, W, H> &other) const
    {
        Matrix<T, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = at(x, y) + other.at(x, y);
            }
        }
        return r;
    }

    Matrix<T, W, H> &operator+=(const Matrix<T, W, H> &other)
    {
        return *this = *this + other;
    }

    Matrix<T, W, H> &operator*=(double x)
    {
        return *this = x * *this;
    }

    Matrix<T, W, H> operator-() const
    {
        Matrix<T, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = -at(x, y);
            }
        }
        return r;
    }

    Matrix<T, W, H> operator-(const Matrix<T, W, H> &other) const
    {
        return *this + (-other);
    }

    template<size_t W_>
    Matrix<T, W_, H> operator*(const Matrix<T, W_, W> &other)
    {
        Matrix<T, W_, H> r;
        for (size_t x = 0; x < W_; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = 0;
                for (size_t i = 0; i < W; i++)
                {
                    r.at(x, y) += at(i, y) * other.at(x, i);
                }
            }
        }
        return r;
    }

    Matrix<T, W, H> operator/(T other)
    {
        Matrix<T, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = at(x, y) / other;
            }
        }
        return r;
    }

    static Matrix<T, W, H> element_by_element_product(const Matrix<T, W, H> &m1, const Matrix<T, W, H> &m2)
    {
        Matrix<T, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = m1.at(x, y) * m2.at(x, y);
            }
        }
        return r;
    }

    friend ostream &operator<<(ostream &s, const Matrix<T, W, H> &m)
    {
        s << m.to_string();
        return s;
    }

    template<class R>
    Matrix<R, W, H> fmap(function<R(T)> f) const
    {
        Matrix<R, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = f(at(x, y));
            }
        }
        return r;
    }

    void iter(function<void(T &)> f)
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                f(at(x, y));
            }
        }
    }

    void enum_iter(const function<void(size_t x, size_t y)> &f)
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                f(x, y);
            }
        }
    }

    template<class R>
    R fold(function<R(R, T)> f, R first_value) const
    {
        R r = first_value;
        for (auto i:arr)
        {
            for (T j:i)
            {
                r = f(r, j);
            }
        }
        return r;
    }

    template<class R>
    Matrix<R, 1, H>
    column_fold(function<Vector<R, H>(Vector<R, H>, Vector<T, H>)> f, Vector<R, H> first_value) const
    {
        Matrix<R, 1, H> r = first_value;
        for (size_t x = 0; x < W; x++)
        {
            r = f(r, column(x));
        }
        return r;
    }

    template<class R>
    RowVector<R, W>
    row_fold(function<RowVector<R, W>(RowVector<R, W>, RowVector<T, W>)> f, RowVector<R, W> first_value) const
    {
        Matrix<T, H, W> t = transpose();
        return t.column_fold(f);
    }

    T max() const
    {
        return fold<T>(function([](T prec_max, T val)
                                { return (val > prec_max) ? val : prec_max; }), at(0, 0));

    }

    T sum() const
    {
        return fold<T>(function([](T acc, T val)
                                { return acc + val; }), 0);
    }

    Vector<T, H> column_sum() const
    {
        return column_fold<T>(function([](Matrix<T, 1, H> acc, Matrix<T, 1, H> val)
                                       { return acc + val; }), {0});
    }

    RowVector<T, W> row_sum() const
    {
        return transpose().column_sum();
    }

    [[nodiscard]] string to_string() const
    {
        array<string, H> lines;
        size_t max_len = 0;

        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                string s = std::to_string(at(x, y));
                lines.at(y) += s;
                if (max_len < lines.at(y).size()) max_len = lines.at(y).size();
            }
            max_len += 2;

            for (size_t y = 0; y < H; y++)
            {
                for (size_t i = 0; i <= (max_len - lines.at(y).size()); i++) lines.at(y) += " ";
            }
        }

        string r;
        for (auto line:lines)
        {
            r += line;
            r.push_back('\n');
        }
        return r;
    }
};

template<class T, size_t W, size_t H>
Matrix<T, W, H> operator*(double x, Matrix<T, W, H> m)
{
    return m.fmap(function([=](T y)
                           { return y * x; }));
}

class MatrixFactory
{
public:
    template<class T, size_t W, size_t H>
    static Matrix<double, W, H> uniform(T val)
    {
        Matrix<T, W, H> r;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r.at(x, y) = val;
            }
        }
        return r;
    }

    template<size_t N>
    static Matrix<double, N, N> diagonal(double val)
    {
        Matrix<double, N, N> r;
        for (size_t x = 0; x < N; x++)
        {
            for (size_t y = 0; y < N; y++)
            {
                r.at(x, y) = (x == y) ? val : 0;
            }
        }

    }

    template<class T, size_t H, size_t W>
    static Matrix<T, H, W> uniform_columns(Vector<T, H> column)
    {
        Matrix<T, H, W> r;
        for (size_t x = 0; x < W; x++)
        {
            r.column(x) = column.column(0);
        }
    }
};

#endif //NEURAL_NETWORK_MATRIX_H
