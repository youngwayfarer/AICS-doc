cmake_minimum_required(VERSION 3.10)
project(MLP_Multiply CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(MLP_SOURCES
    mlp.cpp
)

add_library(mlp_lib STATIC ${MLP_SOURCES})

add_executable(train train.cpp)
target_link_libraries(train mlp_lib)

add_executable(validate validate.cpp)
target_link_libraries(validate mlp_lib)

install(TARGETS train validate
        RUNTIME DESTINATION bin)

install(FILES mlp.h
        DESTINATION include)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mlp_lib OpenMP::OpenMP_CXX)
    target_link_libraries(train OpenMP::OpenMP_CXX)
    target_link_libraries(validate OpenMP::OpenMP_CXX)
    target_compile_definitions(mlp_lib PRIVATE _OPENMP)
    target_compile_definitions(train PRIVATE _OPENMP)
    target_compile_definitions(validate PRIVATE _OPENMP)
    message(STATUS "OpenMP found - enabling parallel processing")
else()
    message(STATUS "OpenMP not found - using serial processing")
endif()

message(STATUS "MLP项目配置:")
message(STATUS "  C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "  编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "  安装前缀: ${CMAKE_INSTALL_PREFIX}")
