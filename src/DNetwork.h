//
// Created by emile on 02/08/2019.
//

#ifndef NEURAL_NETWORK_DNETWORK_H
#define NEURAL_NETWORK_DNETWORK_H

#include "DLayer.h"

template<size_t... sizes>
class D_Network
{

public:

    D_Network() = delete;
};

template<size_t size1, size_t size2>
class D_Network<size1, size2>
{
public:
    D_Layer<size1, size2> d_layer;

    D_Network() = default;

    explicit D_Network(D_Layer<size1, size2> d_layer_) : d_layer(d_layer_)
    {

    }


    D_Network<size1, size2> operator+(D_Network<size1, size2> other)
    {
        return D_Network(d_layer + other.d_layer);
    }


    D_Network<size1, size2> operator+=(D_Network<size1, size2> other)
    {
        d_layer += other.d_layer;
        return *this;
    }


    D_Network<size1, size2> operator*=(double x)
    {
        d_layer *= x;
        return *this;
    }


    D_Network<size1, size2> operator/=(double x)
    {
        d_layer /= x;
        return *this;
    }

    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }
};

template<size_t size1, size_t size2, size_t... sizes>
class D_Network<size1, size2, sizes...>
{
public:
    D_Layer<size1, size2> d_layer;

    D_Network<size2, sizes...> other_layers;

    D_Network() = default;

    explicit D_Network(D_Layer<size1, size2> d_layer_, D_Network<size2, sizes...> other_layers_) : d_layer(d_layer_),
                                                                                                   other_layers(
                                                                                                           other_layers_)
    {

    }

    constexpr static size_t get_number_of_layers()
    {
        return 1 + D_Layer<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if (i == 0)
        {
            return size1;
        } else
        {
            return D_Network<size2, sizes...>::template get_size<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return D_Network<size2, sizes...>::last_size();
    }

    D_Network<size1, size2, sizes...> operator+(D_Network<size1, size2, sizes...> other) const
    {
        return D_Network(d_layer + other.d_layer, other_layers + other.other_layers);
    }


    D_Network<size1, size2, sizes...> &operator+=(D_Network<size1, size2, sizes...> other)
    {

        d_layer += other.d_layer;
        other_layers += other.other_layers;
        return *this;
    }

    D_Network<size1, size2, sizes...> &operator*=(double x)
    {

        d_layer *= x;
        other_layers *= x;
        return *this;
    }

    D_Network<size1, size2, sizes...> &operator/=(double x)
    {

        d_layer /= x;
        other_layers /= x;
        return *this;
    }


};


#endif //NEURAL_NETWORK_DNETWORK_H
