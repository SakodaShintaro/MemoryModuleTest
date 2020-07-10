cmake_minimum_required(VERSION 3.10)
project(MemoryModuleTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_PREFIX_PATH ../../libtorch)
find_package(Torch REQUIRED)


add_executable(MemoryModuleTest
        main.cpp
        differential_neural_computer.cpp
        lstm.cpp
        task.cpp
        )

target_link_libraries(MemoryModuleTest  "${TORCH_LIBRARIES}")