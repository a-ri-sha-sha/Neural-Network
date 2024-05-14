include_directories("../eigen")
include_directories("../EigenRand")
include_directories("../Network")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Tests main.cpp test.cpp)


