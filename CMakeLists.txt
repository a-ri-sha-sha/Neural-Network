cmake_minimum_required(VERSION 3.24)
project(Neural-Network)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(eigen)
add_subdirectory(Network)
add_subdirectory(EigenRand)
add_subdirectory(test)
