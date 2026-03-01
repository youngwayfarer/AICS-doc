# C++ MLP网络预测两个输入的乘积

本项目使用纯C++实现了一个多层感知机(MLP)网络，用于预测两个输入数字的乘积结果。

## 项目结构

```
lab1/
├── mlp.h              # MLP网络类头文件
├── mlp.cpp            # MLP网络类实现
├── train.cpp          # 训练程序
├── validate.cpp       # 验证程序
├── CMakeLists.txt     # CMake构建文件
├── build.sh           # 构建脚本
├── train_data.csv     # 训练数据
├── val_data.csv       # 验证数据
├── README.md          # 实验说明文档
└── test_data.csv      # 测试数据
```

## 环境要求

### 操作系统
- **Ubuntu 20.04或更高版本**（推荐）

> 注意：使用其他系统可能需要对构建脚本做一些修改，或者手动输入命令进行编译，推荐使用 Ubuntu 系统。

### 必需工具
- C++17 或更高版本
- CMake 3.10 或更高版本
- Make 构建工具
- 支持C++17的编译器（GCC 7+等）

在 Ubuntu 上，可以使用以下命令安装相关必要工具：

```bash
sudo apt update
sudo apt install build-essential cmake
```
可以使用以下命令检查是否正确安装：

```bash
gcc --version
make --version
cmake --version
```

### 可选工具
- OpenMP（用于并行化加速）

一般情况下，GCC自带OpenMP支持，无需额外安装。

## 快速开始

### 1. 构建项目

```bash
# 使用构建脚本
bash build.sh
```

### 2. 运行程序

```bash
cd build

# 训练模型
./bin/train

# 验证模型
./bin/validate
```
