include_directories("../eigen")
include_directories("../EigenRand")
include_directories("../Network")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Tests main.cpp test.cpp)
find_package(Eigen3)
target_link_libraries(Tests Eigen3::Eigen)


