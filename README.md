# stack_only_neural_network

A templated neural network. The goal at first was to allocate almost all of the memory in the stack.

It turns out that it is not a very smart idea, since you can only allocated a limited amount of memory on the stack.

It is still possible without much work to make this code stack only : just replace the unsigned_ptr<array<...>> member in Matrix by array<...>. It will segfault if you try to declare and use a big neural network.

This also means that if you want to change the number of layers or their size, you need to recompile.


This project has two main goals :
- Get an understanding of how a neural network works.
- Learn advanced C++ concepts (template metaprogramming, threading, and move semantics).
