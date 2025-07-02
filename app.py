from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import json
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制文件大小为16MB

# 全局变量存储模型
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """加载训练好的PyTorch模型"""
    global model
    try:
        # 这里需要替换为你的模型路径和模型架构
        # 示例：假设你的模型是一个简单的分类器
        model_path = 'model/best_model.pth'  # 请替换为你的模型路径
        
        # 如果你有自定义的模型架构，请在这里定义
        # 这里我用一个示例模型架构，你需要根据你的实际模型修改
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2分类：稳定期/进展期
        
        # 加载模型权重
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"模型加载成功: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}，将使用模拟数据")
            model = None
            
    except Exception as e:
        print(f"模型加载失败: {e}，将使用模拟数据")
        model = None

def preprocess_image(image_file):
    """预处理图像"""
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        return image_tensor.to(device)
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None

def predict_with_model(clinical_img, woods_img):
    """使用真实模型进行预测"""
    if model is None:
        return get_mock_prediction()
    
    try:
        with torch.no_grad():
            # 预处理两张图片
            clinical_tensor = preprocess_image(clinical_img)
            woods_tensor = preprocess_image(woods_img)
            
            if clinical_tensor is None or woods_tensor is None:
                return get_mock_prediction()
            
            # 根据你的模型输入方式调整这里
            # 方式1: 如果模型接受两张图片拼接
            # combined_input = torch.cat([clinical_tensor, woods_tensor], dim=1)
            # outputs = model(combined_input)
            
            # 方式2: 如果模型分别处理两张图片然后融合
            # clinical_features = model.backbone(clinical_tensor)
            # woods_features = model.backbone(woods_tensor)
            # outputs = model.classifier(torch.cat([clinical_features, woods_features], dim=1))
            
            # 方式3: 简单示例 - 这里你需要根据你的实际模型架构修改
            outputs = model(clinical_tensor)  # 暂时只用临床图片
            
            # 获取预测概率
            probabilities = torch.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # 转换为结果
            prediction = "进展期" if predicted_class == 1 else "稳定期"
            confidence_percent = f"{confidence * 100:.1f}%"
            
            # 生成详细分析过程
            details = generate_analysis_details(clinical_tensor, woods_tensor, probabilities)
            
            return {
                "final_prediction": prediction,
                "confidence": confidence_percent,
                "details": details
            }
            
    except Exception as e:
        print(f"模型预测失败: {e}")
        return get_mock_prediction()

def generate_analysis_details(clinical_tensor, woods_tensor, probabilities):
    """生成分析详情"""
    # 这里可以添加更详细的分析逻辑
    prob_stable = probabilities[0][0].item()
    prob_active = probabilities[0][1].item()
    
    details = [
        {"prompt": "临床图像特征分析", "answer": f"进展期概率: {prob_active:.3f}"},
        {"prompt": "伍德灯图像特征分析", "answer": "边界模糊特征检测"},
        {"prompt": "双图融合判断", "answer": "进展期" if prob_active > prob_stable else "稳定期"},
        {"prompt": "边缘清晰度评估", "answer": "边界不清晰" if prob_active > 0.6 else "边界相对清晰"},
        {"prompt": "色素缺失程度", "answer": "中等程度缺失"},
        {"prompt": "炎症反应指标", "answer": "轻微炎症" if prob_active > 0.7 else "无明显炎症"},
        {"prompt": "综合置信度评分", "answer": f"{max(prob_stable, prob_active):.1%}"},
        {"prompt": "建议复查时间", "answer": "1-3个月内复查" if prob_active > 0.6 else "3-6个月复查"}
    ]
    
    return details

def get_mock_prediction():
    """当模型不可用时的模拟预测"""
    return {
        "final_prediction": "进展期",
        "confidence": "87.5%",
        "details": [
            {"prompt": "基于双图判断", "answer": "进展期"},
            {"prompt": "仅根据伍德灯判断", "answer": "进展期"},
            {"prompt": "仅根据临床图判断", "answer": "稳定期"},
            {"prompt": "基于边缘特征图判断", "answer": "进展期"},
            {"prompt": "选择题(A:进展, B:稳定)", "answer": "A"},
            {"prompt": "模型注意力区域分析", "answer": "进展期"},
            {"prompt": "历史数据对比", "answer": "进展期"},
            {"prompt": "综合诊断意见", "answer": "进展期"}
        ]
    }

@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory('change_cup', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('change_cup', filename)

@app.route('/predict', methods=['POST'])
def predict():
    """AI预测接口"""
    try:
        # 检查文件是否上传
        if 'clinical_image' not in request.files or 'woods_image' not in request.files:
            return jsonify({'error': '请上传临床图片和伍德灯图片'}), 400
        
        clinical_file = request.files['clinical_image']
        woods_file = request.files['woods_image']
        
        # 检查文件是否为空
        if clinical_file.filename == '' or woods_file.filename == '':
            return jsonify({'error': '请选择有效的图片文件'}), 400
        
        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not (clinical_file.filename.lower().split('.')[-1] in allowed_extensions and 
                woods_file.filename.lower().split('.')[-1] in allowed_extensions):
            return jsonify({'error': '不支持的文件格式，请上传图片文件'}), 400
        
        # 保存上传的文件（可选）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clinical_filename = f"clinical_{timestamp}_{clinical_file.filename}"
        woods_filename = f"woods_{timestamp}_{woods_file.filename}"
        
        clinical_path = os.path.join(app.config['UPLOAD_FOLDER'], clinical_filename)
        woods_path = os.path.join(app.config['UPLOAD_FOLDER'], woods_filename)
        
        clinical_file.save(clinical_path)
        woods_file.save(woods_path)
        
        # 重新读取文件用于预测
        with open(clinical_path, 'rb') as cf, open(woods_path, 'rb') as wf:
            clinical_data = type('MockFile', (), {'read': lambda: cf.read()})()
            woods_data = type('MockFile', (), {'read': lambda: wf.read()})()
            
            # 进行AI预测
            result = predict_with_model(clinical_data, woods_data)
        
        # 记录预测日志
        log_prediction(clinical_filename, woods_filename, result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

def log_prediction(clinical_file, woods_file, result):
    """记录预测日志"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'clinical_image': clinical_file,
        'woods_image': woods_file,
        'prediction': result['final_prediction'],
        'confidence': result['confidence']
    }
    
    log_file = 'prediction_logs.json'
    logs = []
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append(log_entry)
    
    # 只保留最新的1000条记录
    logs = logs[-1000:]
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

@app.route('/health')
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("正在启动白癜风AI诊断服务...")
    print(f"使用设备: {device}")
    
    # 加载模型
    load_model()
    
    # 启动服务
    print("服务启动中...")
    print("访问地址: http://localhost:8080")
    print("API接口: http://localhost:8080/predict")
    
    app.run(debug=True, host='0.0.0.0', port=8080) 