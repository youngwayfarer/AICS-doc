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
import torchvision as models
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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

## 模型结构的修改

在某些情况下，我们可能需要对预训练模型的结构进行修改，例如更改最后的全连接层以适应不同的分类任务；或者加入一些自定义的层以增强模型的能力。

下面以 ResNet50 为例，在最后的 Bottleneck 块与其之后的全局平均池化层之间，插入一个卷积层、一个批归一化层、一个 Relu 层，来介绍修改模型结构的一种方法。

为了简洁明了，创建一个 class 来表示修改后的模型：

```python
class ResNet50WithExtraConv(nn.Module):
    def __init__(self, extra_conv_channels=128):
        super().__init__()
        original = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = original.conv1
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool
        self.layer1 = original.layer1
        self.layer2 = original.layer2
        self.layer3 = original.layer3
        self.layer4 = original.layer4

        # 新插入的三层
        # 注意：这里插入的卷积层的权重参数为随机生成，并且没有偏置项
        self.extra_conv = nn.Conv2d(2048, extra_conv_channels, 3, 1, 1, bias=False)
        self.extra_bn = nn.BatchNorm2d(extra_conv_channels)
        self.extra_relu = nn.ReLU(inplace=True)

        self.avgpool = original.avgpool
        self.fc = nn.Linear(extra_conv_channels, 1000)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.extra_relu(self.extra_bn(self.extra_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

然后就可以通过这个 class 来对修改后的模型做一些操作。

我们将其保存为 pth、onnx 两种格式的模型文件：

```python
model = ResNet50WithExtraConv()

torch.save(model, 'modified_resnet50_full.pth')
print("完整模型已保存: modified_resnet50_full.pth")

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, (dummy_input, ), 'modified_resnet50.onnx', verbose=True, input_names=['input'], output_names=['output'])
print("ONNX 文件已导出: modified_resnet50.onnx")
```

> 对于 onnx 格式的模型文件，可以通过 [NETRON](https://netron.app/) 来可视化得查看模型结构。

如果需要重新训练模型，以在我们新插入的层中得到合适的参数，而不修改原有层中的参数，可以先冻结原有层：

```python
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.bn1.parameters():
    param.requires_grad = False
for param in model.maxpool.parameters():
    param.requires_grad = False
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False
for param in model.layer3.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = False
for param in model.avgpool.parameters():
    param.requires_grad = False
```

> 如果需要冻结的层数特别多，这部分可以怎样简化？

然后训练即可：

```python
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

# 这里的数据集已经下载到了本地的 ./imagenet 目录下
train_dataset = ImageNet(root='./imagenet', split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

print("训练完成！")
```

训练完成后重新测试模型性能。

## 实验内容

### 5.1 实验任务

#### 5.1.1 必做部分

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

#### 5.1.2 选做部分

根据你选择的模型，尝试修改其中的经典结构，比如增加或者删除一些层，并进行重新训练，训练后与原模型进行性能对比。

你需要在实验报告中说明你修改了哪些结构，并使用 [NETRON](https://netron.app/) 来可视化展示你修改前后结构对比（局部对比即可），并说明你重新训练使用的数据集以及修改前后模型性能对比。

> **注意**： 请确保你选择的训练数据集适合你的硬件配置，你可以选择下载一些数据集的子集或者自己裁剪数据集以满足硬件配置、时间等的要求。

本次实验完成选做部分可以额外拿到 2 分，连同必做部分共拿到 12 分，

### 5.2 提交要求

你需要在实验报告中附上你的代码实现思路、测试数据、运行结果截图或者运行结果处理后的图片，修改后的模型的 pth 文件（如果你完成了选做部分的话），以及对实验现象的分析与总结。
    
请将你的代码文件、测试数据文件、模型输出结果文件（如果存在的话）和实验报告打包到一个 zip ⽂件中（学号_姓名.zip），并将该 zip ⽂件提交到 bb 平台上。

<span style="color:red; font-weight:bold;">提交截⽌时间</span>：北京时间11⽉21⽇ 23:59
