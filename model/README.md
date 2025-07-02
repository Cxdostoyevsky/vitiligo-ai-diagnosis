# 模型文件说明

## 模型文件放置

请将你训练好的PyTorch模型文件放在这个目录下，并命名为 `best_model.pth`

## 模型要求

1. **模型输入**: 
   - 图像尺寸: 224x224
   - 通道数: 3 (RGB)
   - 数据类型: torch.float32
   - 归一化: ImageNet标准 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

2. **模型输出**:
   - 2分类: [稳定期, 进展期]
   - 输出维度: (batch_size, 2)

3. **模型架构**:
   - 当前代码示例使用ResNet50
   - 如果你使用其他架构，请修改 `app.py` 中的 `load_model()` 函数

## 自定义模型架构

如果你的模型架构不是ResNet50，请按以下步骤修改：

1. 在 `app.py` 的 `load_model()` 函数中，替换模型定义部分：

```python
# 替换这部分代码
from torchvision.models import resnet50
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# 改为你的模型架构，例如：
# import your_model_module
# model = your_model_module.YourCustomModel(num_classes=2)
```

2. 如果你的模型需要两张图片作为输入，请修改 `predict_with_model()` 函数中的推理部分。

## 测试你的模型

1. 将模型文件复制到此目录: `model/best_model.pth`
2. 运行服务器: `python app.py`
3. 访问 http://localhost:5000 测试上传功能
4. 检查终端输出确认模型加载成功

## 注意事项

- 确保模型文件完整且可以被PyTorch加载
- 模型应该处于评估模式 (model.eval())
- 如果模型文件很大，考虑使用GPU加速推理 