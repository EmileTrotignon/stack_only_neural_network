cmake_minimum_required(VERSION 3.14)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)


add_executable(nn src/main2.cpp)

add_executable(neural_network src/main.cpp
        src/NeuralNetwork.cpp src/NeuralNetwork.h
        src/Matrix.cpp src/Matrix.h
        src/relevant_math.cpp src/relevant_math.h
        src/RecTuple.cpp src/RecTuple.h
        src/DLayer.cpp src/DLayer.h
        src/DeltaNetwork.cpp src/DeltaNetwork.h
        src/NeuralNetworkInside.cpp src/NeuralNetworkInside.h
        src/NeuralNetworkBase.cpp src/NeuralNetworkBase.h src/NeuralNetworkBpg.cpp src/NeuralNetworkBpg.h src/NeuralNetworkInsideBpg.h src/GenePool.cpp src/GenePool.h)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(neural_network Threads::Threads)


add_executable(unit_test src/unit_test.cpp src/NeuralNetwork.cpp src/NeuralNetwork.h src/Matrix.cpp src/Matrix.h src/relevant_math.cpp src/relevant_math.h src/Matrixtibo.cpp src/Matrixtibo.h)