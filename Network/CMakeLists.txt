
find_package(Eigen3)
add_executable(${PROJECT_NAME} main.cpp net.cpp ActivationFunction.cpp LossFunction.cpp Layer.cpp)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

include_directories("../eigen")
include_directories("../EigenRand")

file(GLOB SOURCES "*.cpp")
