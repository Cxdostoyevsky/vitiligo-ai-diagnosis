# 白癜风AI诊断系统

一个基于PyTorch的白癜风分期AI诊断系统，支持临床皮损照片和伍德灯照片的双图分析。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置模型文件

将训练好的PyTorch模型文件放到 `model/best_model.pth`

> 如果没有模型文件，系统将使用模拟数据运行，你仍然可以测试界面功能

### 3. 启动服务

```bash
python start.py
```

或者直接运行：

```bash
python app.py
```

### 4. 访问系统

打开浏览器，访问：http://localhost:8080

## 📁 项目结构

```
daily/
├── app.py                 # Flask后端API服务
├── start.py              # 启动脚本（推荐使用）
├── requirements.txt      # Python依赖包
├── README.md            # 项目说明
├── change_cup/          # 前端静态文件
│   └── index.html       # 主页面
├── model/               # 模型文件目录
│   ├── README.md        # 模型使用说明
│   └── best_model.pth   # 你的PyTorch模型（需要自己放置）
├── uploads/             # 用户上传的图片存储
└── prediction_logs.json # 预测记录日志
```

## 🔧 功能特性

- ✅ **双图分析**: 支持临床皮损照片 + 伍德灯照片
- ✅ **AI预测**: 自动判断白癜风分期（稳定期/进展期）
- ✅ **置信度显示**: 提供预测的置信度百分比
- ✅ **详细分析**: 展示模型的判断过程和分析步骤
- ✅ **错误处理**: 完善的错误提示和异常处理
- ✅ **日志记录**: 自动记录所有预测结果
- ✅ **响应式设计**: 支持不同设备访问

## 🖥️ 系统要求

- Python 3.7+
- PyTorch 1.12+
- 内存: 至少2GB RAM
- 存储: 至少1GB可用空间
- GPU: 可选，但推荐用于更快的推理速度

## 🔧 自定义模型

### 模型要求

- **输入**: 224x224 RGB图像
- **输出**: 2分类 [稳定期, 进展期]
- **格式**: PyTorch (.pth) 模型文件

### 修改模型架构

如果你的模型不是ResNet50，请修改 `app.py` 中的 `load_model()` 函数：

```python
def load_model():
    global model
    try:
        # 替换为你的模型架构
        model = YourCustomModel(num_classes=2)  # 你的模型
        
        model_path = 'model/best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        model = None
```

### 双图输入模型

如果你的模型需要同时处理两张图片，请修改 `predict_with_model()` 函数：

```python
# 示例：拼接两张图片
combined_input = torch.cat([clinical_tensor, woods_tensor], dim=1)
outputs = model(combined_input)

# 或者：分别提取特征后融合
clinical_features = model.backbone(clinical_tensor)
woods_features = model.backbone(woods_tensor)
outputs = model.classifier(torch.cat([clinical_features, woods_features], dim=1))
```

## 📊 API接口

### 预测接口

**POST** `/predict`

请求参数：
- `clinical_image`: 临床皮损照片文件
- `woods_image`: 伍德灯照片文件

响应格式：
```json
{
  "final_prediction": "进展期",
  "confidence": "87.5%",
  "details": [
    {"prompt": "临床图像特征分析", "answer": "进展期概率: 0.875"},
    {"prompt": "伍德灯图像特征分析", "answer": "边界模糊特征检测"},
    ...
  ]
}
```

### 健康检查接口

**GET** `/health`

响应格式：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "timestamp": "2024-01-20T10:30:00"
}
```

## 🌐 部署到服务器

### 本地部署

系统默认运行在 `localhost:8080`，可以在局域网内访问。

### 云服务器部署

1. **上传代码到服务器**
2. **安装依赖**: `pip install -r requirements.txt`
3. **放置模型文件**: 上传你的 `.pth` 模型到 `model/` 目录
4. **启动服务**: `python start.py`
5. **配置防火墙**: 开放8080端口
6. **域名绑定**: 可选，绑定域名或使用IP访问

### 使用Docker部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "app.py"]
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型架构与代码中定义的一致
   - 查看终端错误信息

2. **依赖包安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用国内镜像: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

3. **GPU不可用**
   - 系统会自动降级到CPU运行
   - 安装GPU版本PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

4. **端口被占用**
   - 修改 `app.py` 中的端口号
   - 或者杀死占用8080端口的进程
   - macOS用户注意：5000端口被AirPlay占用，我们默认使用8080端口

## 📝 更新日志

- **v1.0.0**: 初始版本，支持双图AI诊断
- 支持PyTorch模型加载和推理
- 响应式Web界面
- 完整的错误处理和日志记录

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

如果有任何问题，请查看 `model/README.md` 或提交Issue。 
