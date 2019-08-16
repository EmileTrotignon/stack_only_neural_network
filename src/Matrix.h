//
// Created by emile on 07/06/19.
//

#ifndef NEURAL_NETWORK_MATRIX_H
#define NEURAL_NETWORK_MATRIX_H

#include <iostream>
#include <array>
#include <functional>
#include <memory>

using namespace std;


template<class T, size_t W, size_t H>
class Matrix;

template<class T, size_t H> using Vector = Matrix<T, 1, H>;

template<class T, size_t W> using RowVector = Matrix<T, W, 1>;

template<class T, size_t W, size_t H>
class Matrix
{

//region members

private:
#ifndef MAKE_STACK_ONLY
    unique_ptr<array<array<T, H>, W>> arr;
#else
    array<array<T, H>, W> arr;
#endif

//endregion

//region constructor

public:
    Matrix() : arr(
#ifndef MAKE_STACK_ONLY
            new array<array<T, H>, W>()
#endif
    )
    {
    };

private:

    template<size_t x, size_t y, class R, class... Rs>
    void constructor_helper(R v, Rs... vs)
    {
        at(x, y) = v;
        if constexpr (y != H - 1)
        {
            constructor_helper<x, y + 1, Rs...>(vs...);
        } else if constexpr (x != W - 1)
        {
            constructor_helper<x + 1, 0, Rs...>(vs...);
        }
    }

public:
    template<class... Rs>
    explicit Matrix(Rs... v) : Matrix()
    {
        static_assert(sizeof...(Rs) == W * H);
        constructor_helper<0, 0, Rs...>(v...);
    }

    /*explicit Matrix(const array<array<T, H>, W> &arr_) : Matrix()
    {
        &arr = arr_;
    }*/

    Matrix(const Matrix<T, W, H> &other) : Matrix()
    {
#ifndef MAKE_STACK_ONLY
        std::copy(other.arr->begin(), other.arr->end(), arr->begin());
#else
        std::copy(other.arr.begin(), other.arr.end(), arr.begin());
#endif
    }

#ifndef MAKE_STACK_ONLY
    explicit Matrix(Matrix<T, H, W> &&other) : Matrix()
    {
        arr.swap(other.arr);
        //delete other.arr;
        other.arr = nullptr;
    }

#endif

    explicit Matrix(array<Vector<T, H>, W> am)
    {
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                at(x, y) = am.at(x).at(0, y);
            }
        }
    }
/*
    ~Matrix()
    {
        delete arr;
    }
*/
//endregion

//region operators

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

#ifndef MAKE_STACK_ONLY
    Matrix<T, W, H> &operator=(Matrix<T, W, H> &&other) noexcept
    {
        swap(arr, other.arr);
        //delete other.arr;
        other.arr = nullptr;;
        return *this;
    }

#endif


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

    Matrix<T, W, H> &operator-=(const Matrix<T, W, H> &other)
    {
        return *this = *this - other;
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
    Matrix<T, W_, H> operator*(const Matrix<T, W_, W> &other) const
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

    Matrix<T, W, H> operator/(T other) const
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

    friend ostream &operator<<(ostream &s, const Matrix<T, W, H> &m)
    {
        s << m.to_string();
        return s;
    }

//endregion

//region methods

    T at(size_t x, size_t y) const
    {
#ifndef MAKE_STACK_ONLY
        return arr->at(x).at(y);
#else
        return arr.at(x).at(y);
#endif
    }

    T &at(size_t x, size_t y)
    {
#ifndef MAKE_STACK_ONLY
        return arr->at(x).at(y);
#else
        return arr.at(x).at(y);
#endif
    }

    array<T, H> column(size_t x) const
    {
#ifndef MAKE_STACK_ONLY
        return arr->at(x);
#else
        return arr.at(x);
#endif
    }


    array<T, H> &column(size_t x)
    {
#ifndef MAKE_STACK_ONLY
        return arr->at(x);
#else
        return arr.at(x);
#endif
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

    Matrix<T, H, W> transpose() const
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
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                r = f(r, at(x, y));
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

    [[nodiscard]] tuple<size_t, size_t> max_index() const
    {

        size_t mx = 0;
        size_t my = 0;
        for (size_t x = 0; x < W; x++)
        {
            for (size_t y = 0; y < H; y++)
            {
                if (at(x, y) > at(mx, my))
                {
                    mx = x;
                    my = y;
                }
            }
        }
        return {mx, my};
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

//endregion

};

#endif //NEURAL_NETWORK_MATRIX_H
