include_directories("../eigen")
include_directories("../EigenRand")
include_directories("../Network")

add_executable(Tests ${TEST_SOURCES} ${NET_SOURCES})

file(GLOB NET_SOURCES "../Network/*.cpp")
file(GLOB TEST_SOURCES "*.cpp")
