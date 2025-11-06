# Lab2

## 实验目的

- 掌握编程框架的安装与使用
- 熟悉基本的编程语法与结构
- 部署简单的深度学习模型
- 理解硬件加速的基本原理

## 环境配置

此次实验选择的编程框架为如今流行的深度学习框架 PyTorch，GPU 计算加速使用 NVIDIA 的 CUDA 技术。

下面介绍这两种环境的安装步骤。

### 2.1 安装 CUDA

已经安装 CUDA 的同学可以跳过此部分。

1. **选择合适的 CUDA 版本**：在终端输入 `nvidia-smi`，查看当前 GPU 型号和驱动版本。找到 “CUDA Version” 字段，确认当前驱动支持的最高 CUDA 版本。你也可以访问 [NVIDIA 官网](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) 来确认驱动版本所支持的 CUDA 版本，从而选择合适的 CUDA 版本。

2. **下载 CUDA 安装包**：访问 [NVIDIA CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-toolkit-archive)，选择对应的操作系统、版本和安装方式，下载合适的 CUDA 安装包。

3. **安装 CUDA**：根据下载的安装包类型，按照官方文档的指引进行安装。安装过程中请注意选择合适的安装选项，确保 CUDA 工具包和驱动程序正确安装。

4. **验证是否安装成功**： 安装完成后，打开终端，输入 `nvcc --version`，如果显示 CUDA 版本信息，则表示安装成功。

> 具体安装过程可以参考网上教程，相关资料网上较为丰富。

### 2.2 安装 PyTorch

已经安装 PyTorch 的同学可以跳过此部分。

1. **访问 PyTorch 官网**：打开浏览器，访问 [PyTorch 官方网站](https://pytorch.org/)。

2. **选择安装配置**：在首页的 “Get Started” 部分，选择适合你的操作系统、包管理器（如 pip 或 conda）、Python 版本以及 CUDA 版本。

3. **安装**：根据你的选择，官网会生成相应的安装命令。复制该命令并在终端中运行即可。

4. **验证安装**：安装完成后，打开 Python 交互式环境，输入以下代码验证安装是否成功：

   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

   如果能够正确输出 PyTorch 版本号，并且 `torch.cuda.is_available()` 返回 `True`，则表示安装成功。

> Pytorch 的相关用法可以参考官网的 [Tutorials](https://pytorch.org/tutorials/) 部分，里面有丰富的入门和进阶教程。可以用到哪个学哪个，不需要全部掌握。

## Pytorch 模型部署与推理

本次文档中以 ResNet50 为例，介绍如何加载和使用预训练模型。

首先，导入必要的库，并加载预训练模型。

```python
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()
```

针对选择的模型，选择合适的输入数据（如图片等），并做相应的预处理。


```python
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("test.jpg").convert("RGB")
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0).to(device)
```

> **注意**： 不同的模型可能需要不同的输入尺寸和预处理方式，请参考相应模型的文档进行调整。

然后，将输入数据传入模型，进行推理，最后取出结果，并进行后处理。

```python
with torch.no_grad():
    output = model(input_batch)
_, predicted_idx = torch.max(output, 1)
print(f"Predicted class index: {predicted_idx.item()}")
```

## 实验内容

### 4.1 实验任务

请根据你的电脑硬件情况，选择一个合适的深度学习模型。推荐选择一些轻量级的模型，如 MobileNet、ResNet、YOLO 等。

你可以选择在网上下载相关的模型文件，也可以使用 PyTorch 提供的预训练模型。

针对你选择的模型，完成以下任务：

- 选择合适的输入数据（如图片等），并进行相应的预处理。
- 将输入数据传入模型，在 **CPU**、**GPU** 两种配置下进行推理，取出结果，并进行后处理。
- 记录推理时间，并分析模型在 **CPU**、**GPU** 两种配置下的性能表现。

单个样例即可，如选择 ResNet50 模型，使用一张图片作为输入即可。

在记录推理时间时，需要考虑多次运行取平均值，以减少偶然因素的影响。

注意观察模型在不同硬件配置下的性能差异，以及在同一硬件配置下，多次运行模型推理时间的稳定性（如每次运行时间的差异）。

> **注意**： 请确保你选择的模型和输入数据适合你的硬件配置，避免出现内存不足等问题。

### 4.2 提交要求

你需要在实验报告中附上你的代码实现思路、测试数据、运行结果截图或者运行结果处理后的图片，以及对实验现象的分析与总结。

请将你的代码文件、测试数据文件、模型输出结果文件（如果存在的话）和实验报告打包到一个 zip ⽂件中（学号_姓名.zip），并将该 zip ⽂件提交到 bb 平台上。

<span style="color:red; font-weight:bold;">提交截⽌时间</span>：北京时间11⽉ 21⽇ 23:59
