cmake_minimum_required(VERSION 3.10)
project(MLP_Multiply CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 源文件
set(MLP_SOURCES
    mlp.cpp
    data_generator.cpp
)

# 创建静态库
add_library(mlp_lib STATIC ${MLP_SOURCES})

# 训练程序
add_executable(train train.cpp)
target_link_libraries(train mlp_lib)

# 验证程序
add_executable(validate validate.cpp)
target_link_libraries(validate mlp_lib)

# 安装目标
install(TARGETS train validate
        RUNTIME DESTINATION bin)

# 安装头文件
install(FILES mlp.h data_generator.h
        DESTINATION include)

# 如果使用OpenMP（可选）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(train OpenMP::OpenMP_CXX)
    target_link_libraries(validate OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - enabling parallel processing")
endif()

# 打印配置信息
message(STATUS "MLP项目配置:")
message(STATUS "  C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "  编译器: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "  安装前缀: ${CMAKE_INSTALL_PREFIX}")
