import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 input channels, 16 output channels
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)  # 16 input channels, 8 output channels
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(8 * 32 * 32, 10)  # 假设输入图像大小为 32x32

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

def local_unstructured_pruning_example():
    model = SimpleModel()

    # 对 conv1 层进行 20% 的 L1 非结构化剪枝
    prune.l1_unstructured(module=model.conv1, name='weight', amount=0.2)

    # 对 fc1 层进行 10% 的随机非结构化剪枝
    prune.random_unstructured(module=model.fc1, name='weight', amount=0.1)

    return model

def local_restructured_pruning_example():
    model = SimpleModel()

    # 对 conv2 层进行 30% 的通道剪枝
    prune.ln_structured(module=model.conv2, name='weight', amount=0.3, n=2, dim=0)
    return model

def global_unstructured_pruning_example():
    model = SimpleModel()

    # 定义要剪枝的参数列表（全局剪枝）
    parameters_to_prune = [
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
    ]

    # 对所有参数进行 20% 的全局 L1 非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2
    )

    return model

model = SimpleModel()

print("Before Pruning:")
print(f"Conv1 weight shape: {model.conv1.weight.shape}")
print(f"Conv1 weight sparsity: {float(torch.sum(model.conv1.weight == 0)) / model.conv1.weight.numel():.4f}")
print(f"FC1 weight shape: {model.fc1.weight.shape}")
print(f"FC1 weight sparsity: {float(torch.sum(model.fc1.weight == 0)) / model.fc1.weight.numel():.4f}")

