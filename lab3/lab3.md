# Lab3

## 实验目的

- 了解流行的模型压缩方法
- 掌握模型剪枝和量化的基本原理和实现方法
- 借助 Pytorch 框架实现模型剪枝和量化

## 模型量化

### 2.1 模型量化简介

模型量化是指将神经网络中的浮点数参数和激活值转换为低精度表示（如 8 位整数）的过程。量化可以显著减少模型的存储需求和计算开销，从而提高推理速度和降低功耗，特别适用于资源受限的设备（如移动设备和嵌入式系统）。

模型量化按照激活值的量化参数在推理时是否预先确定并保持不变可以分为：

- **动态量化（Dynamic Quantization）**：在推理时动态地将权重和激活值转换为低精度表示。
- **静态量化（Static Quantization）**：在推理前对模型进行量化，并使用校准数据集来确定激活值的量化参数。

按照量化实施阶段可以分为：

- **量化感知训练（Quantization Aware Training, QAT）**：在训练过程中模拟量化效果，使模型在量化后仍能保持较高的精度。
- **训练后量化（Post-Training Quantization, PTQ）**：在训练完成后对模型进行量化，通常需要较少的计算资源。

Pytorch 提供了丰富的量化工具和 API，支持多种量化方法和配置，方便用户在不同场景下应用量化技术。

上述几种技术可以相互组合使用，本次实验介绍利用 Pytorch 来实现 CPU 端上的静态量化以及训练后量化。

> 有兴趣的同学也可以从网络上了解如何利用 Pytorch 来实现另外两种量化。

### 2.2 借助 Pytorch 实现训练后量化和静态量化

PyTorch 中的 FX 模式是 PyTorch 提供的一套用于模型转换和分析的工具集。它通过符号追踪（Symbolic Tracing）技术，将 Python 函数或 torch.nn.Module 转换为一个可访问、可修改的中间表示（Intermediate Representation, IR），一个 Graph 对象。这个 Graph 本质上是一个计算图，记录了模型执行时的所有操作（如 call_module, call_function, get_attr 等）及其依赖关系。

可以利用 FX 模式来实现量化。

首先需要准备好一个预训练模型，可以使用 Pytorch 提供的预训练模型，也可以使用自己训练好的模型。然后利用 FX 模式对模型进行量化。

然后量化前需要准备相关量化配置：

```python
# fbgemm 是 Pytorch 中用于 x86 CPU 的高性能量化后端，如果是 ARM CPU 则可以使用 qnnpack
qconfig = get_default_qconfig("fbgemm")  # 默认是静态量化
# 重定义 qconfig，使用 MinMaxObserver， 并指定量化数据类型为 torch.quint8 和 torch.qint8，以及指定 per_tensor 量化
qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8))
# 将 qconfig 应用到模型的所有层
qconfig_dict = {
    "": qconfig,
}
```

然后，利用 FX 模式对模型进行量化准备、校准和转换，首先将模型转换成 FX GraphModule：

```python
model2 = copy.deepcopy(model)
model_prepared = prepare_fx(model2, qconfig_dict, example_inputs=(torch.randn(1, 3, 224, 224),))
```

> FX GraphModule 是 PyTorch FX 模式中的一种表示形式，它将计算图封装在一个可调用的模块中，可以打印、修改和执行。

然后，对模型进行校准，这里需要准备一个校准数据集，具体实现可以参考以下函数：

```python
def calib_quant_model(model, calib_dataloader):
    model.eval()
    with torch.inference_mode():
        for inputs, labels in calib_dataloader:
            model(inputs)
```

最后将校准后的模型转换为量化模型：

```python
model_int8 = convert_fx(model_prepared)
```

## 模型剪枝

### 3.1 模型剪枝简介

模型剪枝是指通过移除神经网络中不重要的权重或神经元来减少模型大小和计算复杂度的过程。剪枝可以帮助提高模型的推理速度，降低存储需求，并减少过拟合风险。

模型剪枝的方法主要可以分为：

- **非结构化剪枝（Unstructured Pruning）**：通过移除个别权重来实现剪枝，通常基于权重的绝对值或梯度信息进行选择。
- **结构化剪枝（Structured Pruning）**：通过移除整个神经元、通道或层来实现剪枝，通常基于神经元的重要性评分进行选择。

根据剪枝的位置，模型剪枝可以分为：

- **局部剪枝（Local Pruning）**：在每一层独立地进行剪枝。
- **全局剪枝（Global Pruning）**：在整个模型范围内进行剪枝，考虑所有层的权重分布。

Pytorch 提供了丰富的剪枝工具和 API，支持多种剪枝方法和配置，方便用户在不同场景下应用剪枝技术。

### 3.2 借助 Pytorch 实现模型剪枝

为了便于介绍几种剪枝方法，这里定义一个简单的神经网络如下：

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)channels
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(8 * 32 * 32, 10)  # 假设输入图像大小为 32x32

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x
```

#### 3.2.1 局部非结构化剪枝

可以使用 Pytorch 提供的 `torch.nn.utils.prune` 模块来实现局部非结构化剪枝。

非结构化剪枝以元素为单位进行剪枝，可以通过多种策略来选择要剪枝的权重，例如基于权重的绝对值大小（L1 范数）或随机选择。
下面的代码展示了如何对上面定义的 `SimpleModel` 进行局部非结构化剪枝：

```python
prune.l1_unstructured(module=model.conv1, name='weight', amount=0.2)
prune.random_unstructured(module=model.fc1, name='weight', amount=0.1)
```

这两行代码分别对 `conv1` 层的权重进行 20% 的 L1 非结构化剪枝，以及对 `fc1` 层的权重进行 10% 的随机非结构化剪枝。

可以在剪枝前后通过打印权重的稀疏性来观察剪枝效果：

```python
conv1_sparsity = float(torch.sum(model.conv1.weight == 0)) / model.conv1.weight.numel()
print(f"Conv1 weight sparsity: {conv1_sparsity:.4f}")
fc1_sparsity = float(torch.sum(model.fc1.weight == 0)) / model.fc1.weight.numel()
print(f"FC1 weight sparsity: {fc1_sparsity:.4f}")
```

#### 3.2.2 局部结构化剪枝

结构化剪枝以更大粒度进行剪枝，例如按通道或神经元进行剪枝。下面的代码展示了如何对 `SimpleModel` 进行局部结构化剪枝：

```python
prune.ln_structured(module=model.conv2, name='weight', amount=0.3, n=2, dim=0)
```

这行代码对 `conv2` 层的权重按通道进行 30% 的 L2 结构化剪枝，即在第一个维度（输出通道维度）上，按照 L2 范数选择要剪枝的通道，移除 30% 的通道。

#### 3.2.3 全局非结构化剪枝

全局非结构化剪枝是在整个模型范围内进行剪枝，考虑所有层的权重分布。下面的代码展示了如何对 `SimpleModel` 进行全局非结构化剪枝：

```python
parameters_to_prune = [
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
    ]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )
```

首先定义一个包含所有要剪枝层的列表 `parameters_to_prune`，然后使用 `prune.global_unstructured` 函数对这些参数进行全局非结构化剪枝，移除 20% 的权重。

可以计算剪枝前后模型中剪枝层的总稀疏性来观察剪枝效果：

```python
total_params = 0
zero_params = 0
for module, name in parameters_to_prune:
    total_params += module.weight.numel()
    zero_params += torch.sum(module.weight == 0).item()
global_sparsity = zero_params / total_params
print(f"Global weight sparsity: {global_sparsity:.4f}")
```

## 实验内容

### 4.1 实验任务

结合上次实验的内容，根据自己的电脑硬件情况，选择一个合适的深度学习模型，尝试进行模型剪枝和量化，并测试模型性能。

#### 4.1.1 模型量化

参考[模型量化](#模型量化)部分内容，选择合适的量化方法的组合，对你选择的模型进行量化，并测试 CPU 后端上量化前后的模型性能差异。

#### 4.1.2 模型剪枝

参考[模型剪枝](#模型剪枝)部分内容，选择合适的剪枝方法（局部非结构化剪枝、局部结构化剪枝或全局非结构化剪枝），对你选择的模型进行适当剪枝，测试剪枝前后的模型性能差异。

> 你可能需要在剪枝后对模型进行微调训练，以恢复模型性能。

### 4.2 提交要求

你需要在实验报告中附上你的代码实现思路，在量化、剪枝前后进行模型性能对比（以及指明使用的是什么测试数据集），在剪枝部分，你还需要附上相关层在剪枝前后稀疏度的对比，以及对实验现象的分析与总结。

请将你的代码文件、量化和剪枝后的模型 pth 文件和实验报告打包到一个 zip ⽂件中（学号_姓名.zip），并将该 zip ⽂件提交到 bb 平台上。

<span style="color:red; font-weight:bold;">提交截⽌时间</span>：北京时间11⽉28⽇ 23:59
