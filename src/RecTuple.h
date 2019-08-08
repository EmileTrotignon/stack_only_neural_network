//
// Created by emile on 17/06/19.
//


#ifndef NEURAL_NETWORK_RECTUPLE_H
#define NEURAL_NETWORK_RECTUPLE_H

#include <iostream>
#include <tuple>

template<class... Ts>
class RecTuple
{
public :
    RecTuple() = delete;
};

template<size_t N, typename... Ts> using nth_type =
typename std::tuple_element<N, std::tuple<Ts...>>::type;

template<class T>
class RecTuple<T>
{
public:
    T head;
};

template<class head_T, class... tail_Ts>
class RecTuple<head_T, tail_Ts...>
{
public:
    head_T head;
    RecTuple<tail_Ts...> tail;

};

template<size_t x, class T, class...Ts>
nth_type<x, T, Ts...> get_nth(const RecTuple<T, Ts...> &t)
{
    if constexpr (x == 0) return t.head;
    else return get_nth<x - 1, Ts...>(t.tail);
}

template<size_t x, class T, class...Ts>
void set_nth(RecTuple<T, Ts...> &t, const nth_type<x, T, Ts...> &v)
{
    if constexpr (x == 0) t.head = v;
    else set_nth<x - 1, Ts...>(t.tail, v);
}

#endif //NEURAL_NETWORK_RECTUPLE_H
