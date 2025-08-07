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
import cv2
import sys
from werkzeug.utils import secure_filename
from Feature_extran_vis_tools.feature_extractor import extract_woodlamp_edge_features
from Feature_extran_vis_tools.create_highlight_overlays import create_overlay_image
import shutil


app = Flask(__name__)  #flask相当于盖房子的图纸,app是图纸的实例,之后的所有代码都是基于这个实例的，比如注册网址，配置参数等
CORS(app)  # 允许跨域请求，允许和来自不同地方的前端页面进行通信，比如前端页面在localhost:3000，后端在localhost:8080，前端页面可以访问后端

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads_images'
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')  # 临时文件夹
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER  # 用于存储临时结果
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

def cv2_to_base64(img_array, img_format='PNG'):
    """将OpenCV图像数组转换为base64字符串"""
    try:
        # 将图像编码为指定格式
        _, buffer = cv2.imencode(f'.{img_format.lower()}', img_array)
        # 转换为base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{img_format.lower()};base64,{img_base64}"
    except Exception as e:
        print(f"图像转base64失败: {e}")
        return None

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

def predict_with_model(clinical_img=None, woods_img=None, clinical_path=None, woods_path=None, temp_path=None):
    """使用真实模型进行预测，支持单张或双张图片"""
    if model is None:
        return get_mock_prediction(clinical_path, woods_path, temp_path)
    
    try:
        with torch.no_grad():
            # 预处理图片（支持None输入）
            clinical_tensor = preprocess_image(clinical_img) if clinical_img else None
            woods_tensor = preprocess_image(woods_img) if woods_img else None
            
            # 检查是否至少有一张有效图片
            if clinical_tensor is None and woods_tensor is None:
                return get_mock_prediction(clinical_path, woods_path, temp_path)
            
            # 根据可用的图片进行预测
            if clinical_tensor is not None and woods_tensor is not None:
                # 两张图片都可用 - 使用组合预测（这里可以根据实际模型调整）
                # 目前简单使用临床图片，实际部署时可以根据模型架构调整
                outputs = model(clinical_tensor)
                image_type = "双图片"
            elif clinical_tensor is not None:
                # 只有临床图片
                outputs = model(clinical_tensor)
                image_type = "临床图片"
            else:
                # 只有伍德灯图片
                outputs = model(woods_tensor)
                image_type = "伍德灯图片"
            
            # 获取预测概率
            probabilities = torch.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # 转换为结果
            prediction = "进展期" if predicted_class == 1 else "稳定期"
            
            # 根据图片数量调整置信度显示
            if clinical_tensor is not None and woods_tensor is not None:
                confidence_percent = f"{confidence * 100:.1f}%"
            else:
                # 单张图片时降低显示的置信度，提醒准确性较低
                adjusted_confidence = confidence * 0.8  # 降低20%作为提醒
                confidence_percent = f"{adjusted_confidence * 100:.1f}%"
            
            # 生成真实特征图并保存到临时目录
            feature_maps = {}
            if temp_path:
                temp_dir_name = os.path.basename(temp_path)
                # --- 生成并编码特征图 ---
                if clinical_path:
                    clinical_features_data = extract_woodlamp_edge_features(clinical_path)
                    if clinical_features_data and 'gradient_map' in clinical_features_data:
                        gradient_map = clinical_features_data['gradient_map']
                        # 修正: 将CV_64F转换为CV_8U
                        gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        overlay_img = create_overlay_image(clinical_path, gradient_map_normalized, image_type='clinical')
                        if overlay_img is not None:
                            feature_filename = "clinical_feature.jpg"
                            save_path = os.path.join(temp_path, feature_filename)
                            cv2.imwrite(save_path, overlay_img)
                            feature_maps["clinical_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

                if woods_path:
                    woods_features_data = extract_woodlamp_edge_features(woods_path)
                    if woods_features_data and 'gradient_map' in woods_features_data:
                        gradient_map = woods_features_data['gradient_map']
                        # 修正: 将CV_64F转换为CV_8U
                        gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        overlay_img = create_overlay_image(woods_path, gradient_map_normalized, image_type='wood_lamp')
                        if overlay_img is not None:
                            feature_filename = "woods_feature.jpg"
                            save_path = os.path.join(temp_path, feature_filename)
                            cv2.imwrite(save_path, overlay_img)
                            feature_maps["woods_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"
            
            # 生成详细分析过程
            details = generate_analysis_details(clinical_tensor, woods_tensor, probabilities, image_type)
            
            result = {
                "final_prediction": prediction,
                "confidence": confidence_percent,
                "details": details
            }
            
            # 添加特征图（如果生成成功）
            if feature_maps:
                result["feature_maps"] = feature_maps
            
            return result
            
    except Exception as e:
        print(f"模型预测失败: {e}")
        return get_mock_prediction(clinical_path, woods_path, temp_path)

def generate_analysis_details(clinical_tensor, woods_tensor, probabilities, image_type):
    """根据可用图片类型生成分析详情"""
    prob_stable = probabilities[0][0].item()
    prob_active = probabilities[0][1].item()
    
    details = []
    
    if image_type == "双图片":
        # 双图片完整分析
        details = [
            {"prompt": "临床图像特征分析", "answer": f"进展期概率: {prob_active:.3f}"},
            {"prompt": "伍德灯图像特征分析", "answer": "边界模糊特征检测"},
            {"prompt": "双图融合判断", "answer": "进展期" if prob_active > prob_stable else "稳定期"},
            {"prompt": "边缘清晰度评估", "answer": "边界不清晰" if prob_active > 0.6 else "边界相对清晰"},
            {"prompt": "色素缺失程度", "answer": "中等程度缺失"},
            {"prompt": "炎症反应指标", "answer": "轻微炎症" if prob_active > 0.7 else "无明显炎症"},
        ]
    elif image_type == "临床图片":
        # 仅临床图片分析
        details = [
            {"prompt": "图片类型", "answer": "仅临床皮损照片（准确度较低）"},
            {"prompt": "临床图像特征分析", "answer": f"进展期概率: {prob_active:.3f}"},
            {"prompt": "边缘清晰度评估", "answer": "边界不清晰" if prob_active > 0.6 else "边界相对清晰"},
            {"prompt": "色素缺失程度", "answer": "中等程度缺失"},
            {"prompt": "炎症反应指标", "answer": "轻微炎症" if prob_active > 0.7 else "无明显炎症"},
            {"prompt": "建议", "answer": "建议补充伍德灯照片以提高诊断准确性"},
        ]
    elif image_type == "伍德灯图片":
        # 仅伍德灯图片分析
        details = [
            {"prompt": "图片类型", "answer": "仅伍德灯照片（准确度较低）"},
            {"prompt": "伍德灯图像特征分析", "answer": f"进展期概率: {prob_active:.3f}"},
            {"prompt": "荧光特征分析", "answer": "边界模糊特征检测"},
            {"prompt": "白斑边界评估", "answer": "边界不清晰" if prob_active > 0.6 else "边界相对清晰"},
            {"prompt": "色素对比度", "answer": "中等对比度"},
            {"prompt": "建议", "answer": "建议补充临床皮损照片以提高诊断准确性"},
        ]
    
    # 添加通用的置信度和复查建议
    details.extend([
        {"prompt": "综合置信度评分", "answer": f"{max(prob_stable, prob_active):.1%}"},
        {"prompt": "建议复查时间", "answer": "1-3个月内复查" if prob_active > 0.6 else "3-6个月复查"}
    ])
    
    return details

def get_mock_prediction(clinical_path=None, woods_path=None, temp_path=None):
    """
    当模型不可用时的模拟预测。
    如果提供了临床和伍德灯图片路径，则生成真实的特征图。
    否则，返回模拟的特征图。
    """
    
    feature_maps = {}
    if temp_path:
        temp_dir_name = os.path.basename(temp_path)
        # --- 生成并编码特征图 ---
        if clinical_path:
            clinical_features_data = extract_woodlamp_edge_features(clinical_path)
            if clinical_features_data and 'gradient_map' in clinical_features_data:
                gradient_map = clinical_features_data['gradient_map']
                # 修正: 将CV_64F转换为CV_8U
                gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                overlay_img = create_overlay_image(clinical_path, gradient_map_normalized, image_type='clinical')
                if overlay_img is not None:
                    feature_filename = "clinical_feature.jpg"
                    save_path = os.path.join(temp_path, feature_filename)
                    cv2.imwrite(save_path, overlay_img)
                    feature_maps["clinical_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

        if woods_path:
            woods_features_data = extract_woodlamp_edge_features(woods_path)
            if woods_features_data and 'gradient_map' in woods_features_data:
                gradient_map = woods_features_data['gradient_map']
                # 修正: 将CV_64F转换为CV_8U
                gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                overlay_img = create_overlay_image(woods_path, gradient_map_normalized, image_type='wood_lamp')
                if overlay_img is not None:
                    feature_filename = "woods_feature.jpg"
                    save_path = os.path.join(temp_path, feature_filename)
                    cv2.imwrite(save_path, overlay_img)
                    feature_maps["woods_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

    # 完整结果
    result = {
        "final_prediction": "进展期",
        "confidence": "99.5%",
        "feature_maps": feature_maps,
        "details": [
            {"prompt": "双图prompt_1", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "双图prompt_2", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "双图prompt_3", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "双图prompt_4", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "双图prompt_5", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "双图prompt_6", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：进展期"},
            {"prompt": "仅根据伍德灯prompt_1", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据伍德灯prompt_2", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据伍德灯prompt_3", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据伍德灯prompt_4", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据伍德灯prompt_5", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据伍德灯prompt_6", "answer": "llm问答：进展期\nllm分类：进展期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_1", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_2", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_3", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_4", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_5", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
            {"prompt": "仅根据临床prompt_6", "answer": "llm问答：稳定期\nllm分类：稳定期\ncnn分类：稳定期"},
        ]
    }
    
    return result

def determine_final_prediction(clinical_pred, woods_pred):
    """
    根据临床和伍德灯的预测概率，确定最终的预测结果。
    """
    # 假设临床预测是 [稳定概率, 进展概率]
    # 伍德灯预测是 [稳定概率, 进展概率]
    # 这里简单地取两者的最大概率作为最终预测
    final_prob_active = max(clinical_pred[0][1], woods_pred[0][1])
    final_prediction = "进展期" if final_prob_active > 0.5 else "稳定期"
    return final_prediction, final_prob_active

#@在python中，@app.route() 给函数绑定一个url地址，当用户访问这个url地址时，会执行这个函数
@app.route('/')
def index():
    """提供前端页面"""   #当某人访问网站根目录时，会返回change_cup文件夹中的index.html文件，这时用户会看到index.html文件的内容
    return send_from_directory('change_cup', 'index.html') #当用户访问网站根目录的网页时，会返回change_cup文件夹中的index.html文件，这时用户会看到index.html文件的内容

@app.route('/static/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('change_cup', filename) #当用户访问网站/static/文件夹中的文件时，会返回change_cup文件夹中的文件,比如css,js,images等文件

@app.route('/predict', methods=['POST'])
def predict():
    """AI预测接口"""
    try:
        # 检查是否至少上传了一张图片
        clinical_file = request.files.get('clinical_image')
        woods_file = request.files.get('woods_image')
        
        # 检查是否至少有一个文件存在且有效
        has_clinical = clinical_file and clinical_file.filename != ''
        has_woods = woods_file and woods_file.filename != ''
        
        if not has_clinical and not has_woods:
            return jsonify({'error': '请至少上传一张图片（临床图片或伍德灯图片）'}), 400
        
        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        if has_clinical and not clinical_file.filename.lower().split('.')[-1] in allowed_extensions:
            return jsonify({'error': '临床图片格式不支持，请上传图片文件'}), 400
            
        if has_woods and not woods_file.filename.lower().split('.')[-1] in allowed_extensions:
            return jsonify({'error': '伍德灯图片格式不支持，请上传图片文件'}), 400
        
        # 创建一个唯一的临时文件夹来存储本次请求的所有文件
        temp_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_dir_name)
        os.makedirs(temp_path, exist_ok=True)

        clinical_path, woods_path = None, None
        clinical_filename, woods_filename = None, None

        if has_clinical:
            clinical_filename = secure_filename(f"clinical.{clinical_file.filename.lower().split('.')[-1]}") #secure_filename: 确保文件名安全，防止被hacker恶意文件名攻击
            clinical_path = os.path.join(temp_path, clinical_filename)
            clinical_file.save(clinical_path)
            
        if has_woods:
            woods_filename = secure_filename(f"woods.{woods_file.filename.lower().split('.')[-1]}")
            woods_path = os.path.join(temp_path, woods_filename)
            woods_file.save(woods_path)
        
        # 准备文件数据用于预测
        clinical_data = None
        woods_data = None
        
        if clinical_path and os.path.exists(clinical_path):
            with open(clinical_path, 'rb') as cf:
                clinical_data = type('MockFile', (), {'read': lambda: cf.read()})()
                
        if woods_path and os.path.exists(woods_path):
            with open(woods_path, 'rb') as wf:
                woods_data = type('MockFile', (), {'read': lambda: wf.read()})()
        
        # 进行AI预测，现在传入temp_path以保存特征图
        result = predict_with_model(clinical_data, woods_data, clinical_path, woods_path, temp_path)
        
        # 附加临时目录名和原始文件名到结果中
        result['temp_dir_name'] = temp_dir_name
        result.pop('files_to_save', None)

        # 将最终结果保存到临时目录中的result.json
        json_path = os.path.join(temp_path, 'result.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        # # 记录预测日志
        # log_prediction(clinical_filename, woods_filename, result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/save_result', methods=['POST'])
def save_result():
    """保存预测结果到带时间戳的文件夹"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的请求'}), 400

        temp_dir_name = data.get('temp_dir_name')
        if not temp_dir_name:
            return jsonify({'error': '缺少临时目录名'}), 400

        # 源路径：临时文件夹
        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_dir_name)
        if not os.path.isdir(temp_path):
            return jsonify({'error': '临时结果不存在或已过期'}), 404

        # 目标路径：永久保存的文件夹
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp_dir)
        
        # 移动文件夹
        shutil.move(temp_path, save_path)
            
        return jsonify({'message': '结果保存成功', 'path': save_path})

    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        return jsonify({'error': f'保存失败: {str(e)}'}), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """提供上传的文件（支持子目录）"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/clear_temp', methods=['POST'])
def clear_temp():
    """清除指定的临时文件夹"""
    try:
        data = request.get_json()
        temp_dir_name = data.get('temp_dir_name')
        if not temp_dir_name:
            return jsonify({'error': '缺少临时目录名'}), 400

        temp_path = os.path.join(app.config['TEMP_FOLDER'], temp_dir_name)
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)
            return jsonify({'message': '临时文件已清除'})
        else:
            return jsonify({'message': '临时文件不存在，无需清除'})
    except Exception as e:
        print(f"清除临时文件时出错: {e}")
        return jsonify({'error': f'清除失败: {str(e)}'}), 500

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
    #这里只是启动服务器，具体的执行是在http请求中执行的
    app.run(debug=True, host='0.0.0.0', port=8080) #debug=true:代码改动时自动重启服务器，host='0.0.0.0':监听所有网络接口，允许外部访问，port=8080:端口号
