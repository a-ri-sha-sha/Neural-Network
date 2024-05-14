
find_package(Eigen3)
add_library(${PROJECT_NAME} net.cpp ActivationFunction.cpp LossFunction.cpp Layer.cpp Except.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(../eigen)
include_directories(../EigenRand)
