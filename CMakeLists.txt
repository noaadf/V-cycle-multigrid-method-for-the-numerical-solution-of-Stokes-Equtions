cmake_minimum_required(VERSION 3.20)

project(final)

set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

find_package(Eigen3 REQUIRED)

# 定义目标 main
add_executable(main main.cpp utils.cpp utils.h)

# 为 Clang 编译器设置 OpenMP 选项
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(main PRIVATE -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/19.1.7/include)
    target_link_options(main PRIVATE -L/opt/homebrew/Cellar/libomp/19.1.7/lib -lomp)
endif()

# 链接 Eigen3 库
target_link_libraries(main Eigen3::Eigen)
