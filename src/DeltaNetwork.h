//
// Created by emile on 02/08/2019.
//

#ifndef NEURAL_NETWORK_DELTANETWORK_H
#define NEURAL_NETWORK_DELTANETWORK_H

#include "DLayer.h"

template<size_t... sizes>
class DeltaNetwork
{
public:
    DeltaNetwork() = delete;
};

template<size_t size1, size_t size2>
class DeltaNetwork<size1, size2>
{

//region constexpr static

public:
    constexpr static size_t get_number_of_layers()
    {
        return 1;
    };

    constexpr static size_t last_size()
    {
        return size2;
    }

//endregion

//region members

public:
    DeltaLayer<size1, size2> d_layer;

//endregion

//region constructors

public:
    DeltaNetwork() = default;

    explicit DeltaNetwork(DeltaLayer<size1, size2> d_layer_) : d_layer(d_layer_)
    {

    }

//endregion

//region operators

public:
    DeltaNetwork<size1, size2> operator+(const DeltaNetwork<size1, size2> &other)
    {
        return DeltaNetwork(d_layer + other.d_layer);
    }


    DeltaNetwork<size1, size2> operator+=(const DeltaNetwork<size1, size2> &other)
    {
        d_layer += other.d_layer;
        return *this;
    }


    DeltaNetwork<size1, size2> operator*=(double x)
    {
        d_layer *= x;
        return *this;
    }


    DeltaNetwork<size1, size2> operator/=(double x)
    {
        d_layer /= x;
        return *this;
    }

//endregions

};

template<size_t size1, size_t size2, size_t... sizes>
class DeltaNetwork<size1, size2, sizes...>
{

//region members

public:
    DeltaLayer<size1, size2> d_layer;

    DeltaNetwork<size2, sizes...> other_layers;

//endregion

//region constructors

    DeltaNetwork() = default;

    explicit DeltaNetwork(DeltaLayer<size1, size2> d_layer_, DeltaNetwork<size2, sizes...> other_layers_) : d_layer(
            d_layer_),
                                                                                                            other_layers(
                                                                                                           other_layers_)
    {

    }

//endregion

//region operators

public:
    DeltaNetwork<size1, size2, sizes...> operator+(DeltaNetwork<size1, size2, sizes...> other) const
    {
        return DeltaNetwork(d_layer + other.d_layer, other_layers + other.other_layers);
    }

    DeltaNetwork<size1, size2, sizes...> &operator+=(DeltaNetwork<size1, size2, sizes...> other)
    {

        d_layer += other.d_layer;
        other_layers += other.other_layers;
        return *this;
    }

    DeltaNetwork<size1, size2, sizes...> &operator*=(double x)
    {

        d_layer *= x;
        other_layers *= x;
        return *this;
    }

    DeltaNetwork<size1, size2, sizes...> &operator/=(double x)
    {

        d_layer /= x;
        other_layers /= x;
        return *this;
    }

//endregion

//region constexpr static

public:
    constexpr static size_t get_number_of_layers()
    {
        return 2 + sizeof...(sizes);// + DeltaLayer<size2, sizes...>::get_number_of_layers();
    }

    template<size_t i>
    constexpr static size_t get_size()
    {
        if constexpr(i == 0)
        {
            return size1;
        } else
        {
            return DeltaNetwork<size2, sizes...>::template DeltaNetwork<i - 1>();
        }
    }

    constexpr static size_t last_size()
    {
        return DeltaNetwork<size2, sizes...>::last_size();
    }

//endregion

};


#endif //NEURAL_NETWORK_DELTANETWORK_H
