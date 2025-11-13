import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 假设你已创建了修改后的模型
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

        # 新增卷积层
        self.extra_conv = nn.Conv2d(2048, extra_conv_channels, 3, 1, 1, bias=False)
        self.extra_bn = nn.BatchNorm2d(extra_conv_channels)
        self.extra_relu = nn.ReLU(inplace=True)

        # 调整分类头
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

# 创建并初始化模型
model = ResNet50WithExtraConv()

original = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# 导出原始 ResNet-50 为 ONNX
dummy_input = torch.randn(1, 3, 224, 224)  # 示例输入，匹配模型输入尺寸
torch.onnx.export(original, (dummy_input,), 'original_resnet50.onnx', verbose=True, input_names=['input'], output_names=['output'])
print("✅ 原始 ResNet-50 ONNX 文件已导出: original_resnet50.onnx")

torch.save(model, 'modified_resnet50_full.pth')
print("✅ 完整模型已保存: modified_resnet50_full.pth")

dummy_input = torch.randn(1, 3, 224, 224)  # 示例输入，匹配模型输入尺寸
torch.onnx.export(model, (dummy_input, ), 'modified_resnet50.onnx', verbose=True, input_names=['input'], output_names=['output'])
print("✅ ONNX 文件已导出: modified_resnet50.onnx")