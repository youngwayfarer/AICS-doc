#!/bin/bash

# MLP项目构建脚本

echo "=== MLP项目构建脚本 ==="
echo "用于构建和运行MLP网络项目"
echo "========================================"

# 检查是否安装了必要的工具
check_dependencies() {
    echo "检查依赖项..."
    
    if ! command -v cmake &> /dev/null; then
        echo "错误: 未找到cmake，请先安装cmake"
        exit 1
    fi
    
    if ! command -v make &> /dev/null; then
        echo "错误: 未找到make，请先安装make"
        exit 1
    fi
    
    echo "✓ 依赖项检查通过"
}

# 创建构建目录
create_build_dir() {
    echo "创建构建目录..."
    if [ -d "build" ]; then
        echo "清理旧的构建目录..."
        rm -rf build
    fi
    mkdir build
    cd build
}

# 配置项目
configure_project() {
    echo "配置项目..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    if [ $? -ne 0 ]; then
        echo "错误: CMake配置失败"
        exit 1
    fi
}

# 编译项目
build_project() {
    echo "编译项目..."
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "错误: 编译失败"
        exit 1
    fi
    echo "✓ 编译完成"
}

# 主函数
main() {
    check_dependencies
    create_build_dir
    configure_project
    build_project
    
    echo ""
    echo "构建完成！"
    echo "可执行文件位置:"
    echo "  - 训练程序: build/bin/train"
    echo "  - 验证程序: build/bin/validate"
    echo ""
    echo "使用方法:"
    echo "  cd build"
    echo "  ./bin/train      # 训练模型"
    echo "  ./bin/validate   # 验证模型"
}

# 运行主函数
main "$@"
