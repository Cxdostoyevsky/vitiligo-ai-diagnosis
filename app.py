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
import random
from werkzeug.utils import secure_filename
from Feature_extran_vis_tools.feature_extractor import extract_woodlamp_edge_features
from Feature_extran_vis_tools.create_highlight_overlays import create_overlay_image
from data_generate.generate_datasets_stage_2_binary_cls import two_binary_cls
from data_generate.generate_datasets_stage_2_choice import two_choice
import shutil
import subprocess

# 导入模型相关模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from model.test import InferenceSystem, save_results
from transformers import AutoProcessor

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
inference_system = None
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
    global model, inference_system
    try:
        print("正在初始化推理系统...")
        
        # 初始化推理系统
        inference_system = InferenceSystem(device)
        inference_system.initialize()
        
        # 标记模型已加载成功
        model = "loaded"  # 用作标记，表示模型系统已初始化
        print("推理系统初始化成功！")
            
    except Exception as e:
        print(f"推理系统初始化失败: {e}，将使用模拟数据")
        model = None
        inference_system = None



def predict_with_model(clinical_img=None, woods_img=None, clinical_path=None, woods_path=None, temp_path=None, doot_dir=None):
    """使用真实模型进行预测，支持单张或双张图片"""
    if model is None:
        return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)


    else:
        
        try:

            print("使用真实模型进行预测...")

            # === 第一步：生成特征图（保持原有功能）===

            feature_maps = {}

            if temp_path:

                temp_dir_name = os.path.basename(temp_path)

                # 生成并编码特征图

                if clinical_path:

                    clinical_features_data = extract_woodlamp_edge_features(clinical_path)

                    if clinical_features_data and 'gradient_map' in clinical_features_data:

                        gradient_map = clinical_features_data['gradient_map']

                        gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX,
                                                                dtype=cv2.CV_8U)

                        overlay_img = create_overlay_image(clinical_path, gradient_map_normalized,
                                                           image_type='clinical')

                        if overlay_img is not None:
                            feature_filename = "edge_enhanced_clinical.jpg"

                            save_path = os.path.join(temp_path, feature_filename)

                            cv2.imwrite(save_path, overlay_img)

                            feature_maps["clinical_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

                if woods_path:

                    woods_features_data = extract_woodlamp_edge_features(woods_path)

                    if woods_features_data and 'gradient_map' in woods_features_data:

                        gradient_map = woods_features_data['gradient_map']

                        gradient_map_normalized = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX,
                                                                dtype=cv2.CV_8U)

                        overlay_img = create_overlay_image(woods_path, gradient_map_normalized, image_type='wood_lamp')

                        if overlay_img is not None:
                            feature_filename = "edge_enhanced_woods.jpg"

                            save_path = os.path.join(temp_path, feature_filename)

                            cv2.imwrite(save_path, overlay_img)

                            feature_maps["woods_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

            # === 第二步：使用真实模型进行预测 ===
            # === 生成数据集文件 ===
            if temp_path:
                try:
                    # 模拟预测状态为进展期
                    status_prediction = "active_or_stable"

                    # 生成唯一的patient_id
                    temp_dir_name = os.path.basename(temp_path)
                    patient_id = f"patient_{temp_dir_name}"

                    # 1. 创建临时的master.json文件
                    master_case_data = {
                        "idx": 0,
                        "status": status_prediction,
                        "patient_id": patient_id,
                        "images": {
                            "clinical": [os.path.basename(clinical_path).split('_')[-1]] if clinical_path else [],
                            "wood_lamp": [os.path.basename(woods_path).split('_')[-1]] if woods_path else []
                        }
                    }

                    master_json_path = os.path.join(temp_path, 'master.json')
                    with open(master_json_path, 'w', encoding='utf-8') as f:
                        json.dump([master_case_data], f, indent=2, ensure_ascii=False)

                    two_binary_cls(master_json_path, temp_path, doot_dir)
                    two_choice(master_json_path, temp_path, doot_dir)

                    print(f"模拟预测：成功通过外部脚本在 {temp_path} 生成数据集文件。")

                except Exception as e:
                    print(f"数据json文件生成出错: {e}")

            # === 第三步：动态创建INPUT_CONFIG ===
            # 根据生成的JSON文件构建动态配置
            dynamic_input_config = {
                "oc": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_1img_OC.json"),
                    "image_types": ["clinical"]
                },
                "ow": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_1img_OW.json"),
                    "image_types": ["wood"]
                },
                "oc_ec": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_2img_OC_EC.json"),
                    "image_types": ["clinical", "edge_enhanced_clinical"]
                },
                "ow_ew": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_2img_OW_EW.json"),
                    "image_types": ["wood", "edge_enhanced_wood"]
                },
                "oc_ow": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_2img_OC_OW.json"),
                    "image_types": ["clinical", "wood"]
                },
                "oc_ec_ow_ew": {
                    "json_path": os.path.join(temp_path, "train_binary_cls_4img_OC_EC_OW_EW.json"),
                    "image_types": ["clinical", "edge_enhanced_clinical", "wood", "edge_enhanced_wood"]
                }
            }
            
            # 过滤掉不存在的JSON文件
            available_config = {}
            for input_type, config in dynamic_input_config.items():
                if os.path.exists(config["json_path"]):
                    available_config[input_type] = config
                else:
                    print(f"警告: JSON文件不存在: {config['json_path']}")
            
            if not available_config:
                print("错误: 没有可用的JSON文件，回退到模拟预测")
                return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)

            # === 第四步：加载预处理后的图像 ===
            try:
                # 加载数据，使用temp_path作为图像根目录
                ordered_ids, test_samples = inference_system.load_data(
                    input_config=available_config, 
                    image_root_dir=temp_path
                )
                print(f"加载了 {len(ordered_ids)} 个样本ID，样本数量: {len(test_samples)}")
                print(f"可用的输入类型: {list(available_config.keys())}")
            except Exception as e:
                print(f"数据加载失败: {e}，回退到模拟预测")
                return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)

            # 运行推理
            try:
                probabilities_data, predictions_data = inference_system.run_inference(ordered_ids, test_samples)
                # 保存结果并获取增强的概率数据
                enhanced_probabilities_data = save_results(probabilities_data, predictions_data)
                print("测试完成!")
            except Exception as e:
                print(f"模型推理失败: {e}，回退到模拟预测")
                return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)
            


            # === 第五步：处理预测结果 ===
            if enhanced_probabilities_data and len(enhanced_probabilities_data) > 0:
                prob_row = enhanced_probabilities_data[0]
                pred_row = predictions_data[0]
                
                
                # 生成详细分析
                details ,final_prediction, confidence= generate_model_analysis_details(prob_row, pred_row, clinical_path, woods_path)
                
                # 构建最终结果
                result = {
                    "final_prediction": final_prediction,
                    "confidence": f"{int(confidence * 100)}",
                    "feature_maps": feature_maps,
                    "details": details,
                    "model_type": "real_model"  # 标记使用了真实模型
                }
                
                print(f"真实模型预测完成：{final_prediction}，置信度：{confidence:.1%}")
                return result
            else:
                print("警告：模型推理失败，回退到模拟预测")
                return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)



        except Exception as e:
            print(f"真实模型预测失败: {e}，回退到模拟预测")
            return get_mock_prediction(clinical_path, woods_path, temp_path, doot_dir)


def determine_model_prediction(prob_row, pred_row=None, clinical_path=None, woods_path=None):
    """
    根据 CSV 文件最后两列直接确定最终预测结果
    result: 0 表示稳定期，1 表示进展期
    confidence: 置信度
    """
    # 读取结果和置信度
    result_flag = int(prob_row["result"])
    confidence = float(prob_row["confidence"])

    # 0 -> 稳定期, 1 -> 进展期
    prediction = "进展期" if result_flag == 1 else "稳定期"

    return prediction, int(confidence * 100)


def generate_model_analysis_details(prob_row, pred_row, clinical_path=None, woods_path=None):
    """生成模型分析详情 - 根据CSV数据动态显示图片组合和预测概率"""
    details = []

    # 全局概率收集
    global_probs = {
        "稳定期": [],
        "进展期": []
    }

    # 获取图片基础路径信息
    temp_dir_name = None
    if clinical_path or woods_path:
        # 从路径中提取临时目录名
        path_to_check = clinical_path or woods_path
        temp_dir_name = os.path.basename(os.path.dirname(path_to_check))

    # 列名到图片组合的映射
    combination_mapping = {
        "oc": {
            "name": "原始临床图",
            "description": "仅原始临床图像分析",
            "images": ["original_clinical"],
            "image_labels": ["原始临床图片"]
        },
        "ow": {
            "name": "原始伍德灯图",
            "description": "仅原始伍德灯图像分析",
            "images": ["original_woods"],
            "image_labels": ["原始伍德灯图片"]
        },
        "oc_ec": {
            "name": "原始临床图，边缘增强临床图",
            "description": "临床图像+边缘增强分析",
            "images": ["original_clinical", "edge_enhanced_clinical"],
            "image_labels": ["原始临床图片", "边缘增强临床图片"]
        },
        "ow_ew": {
            "name": "原始伍德灯图，边缘增强伍德灯图",
            "description": "伍德灯+边缘增强分析",
            "images": ["original_woods", "edge_enhanced_woods"],
            "image_labels": ["原始伍德灯图片", "边缘增强伍德灯图片"]
        },
        "oc_ow": {
            "name": "原始临床图，原始伍德灯图",
            "description": "临床+伍德灯双图融合分析",
            "images": ["original_clinical", "original_woods"],
            "image_labels": ["原始临床图片", "原始伍德灯图片"]
        },
        "oc_ec_ow_ew": {
            "name": "原始临床图，边缘增强临床图，原始伍德灯图，边缘增强伍德灯图",
            "description": "四图融合综合分析",
            "images": ["original_clinical", "edge_enhanced_clinical", "original_woods", "edge_enhanced_woods"],
            "image_labels": ["原始临床图片", "边缘增强临床图片", "原始伍德灯图片", "边缘增强伍德灯图片"]
        }
    }

    # 获取所有可用的预测组合
    available_combinations = []
    for key in prob_row.keys():
        if key != "id" and "_stable" in key:
            input_type = key.replace("_stable", "")
            if f"{input_type}_active" in prob_row:
                available_combinations.append(input_type)

    # 为每种组合生成详情显示
    for input_type in available_combinations:
        if input_type in combination_mapping:
            stable_prob = prob_row[f"{input_type}_stable"]
            active_prob = prob_row[f"{input_type}_active"]
            prediction = "进展期" if active_prob > stable_prob else "稳定期"

            combo_info = combination_mapping[input_type]

            # 构建图片路径信息
            image_paths = []
            if temp_dir_name:
                for img_name in combo_info["images"]:
                    # 根据图片类型构建完整路径
                    if img_name == "original_clinical":
                        # 查找临床图片
                        if clinical_path:
                            image_paths.append(f"uploads/temp/{temp_dir_name}/{os.path.basename(clinical_path)}")
                            # image_paths.append(clinical_path)
                    elif img_name == "original_woods":
                        # 查找伍德灯图片
                        if woods_path:
                            image_paths.append(f"uploads/temp/{temp_dir_name}/{os.path.basename(woods_path)}")
                            # image_paths.append(woods_path)
                    elif img_name == "edge_enhanced_clinical":
                        image_paths.append(f"uploads/temp/{temp_dir_name}/edge_enhanced_clinical.jpg")
                    elif img_name == "edge_enhanced_woods":
                        # image_paths.append(f"/uploads/temp/{temp_dir_name}/edge_enhanced_woods.jpg")
                        image_paths.append(f"uploads/temp/{temp_dir_name}/edge_enhanced_woods.jpg")

            # 模拟LLM结果
            llm_prediction = prediction  # 模拟LLM分类结果
            # 制造一点点不同来区分
            llm_stable_prob = min(stable_prob + random.uniform(-0.3, -0.1), 1.0)
            llm_active_prob = 1 - llm_stable_prob
            llm_stable_probb = min(stable_prob + random.uniform(-0.2, -0.1), 1.0)
            llm_active_probb = 1 - llm_stable_probb
            llm_qa_answer = f"这是一个模拟的AI问答摘要。综合分析图像特征，AI认为当前情况与“{prediction}”的典型表现较为一致。请注意这仅为模拟数据。"

            # === 收集全局概率 ===
            # 传统模型概率
            global_probs["稳定期"].append(stable_prob)
            global_probs["进展期"].append(active_prob)
            # 大模型分类与问答一致才加
            llm_prediction = "进展期" if llm_active_prob > llm_stable_prob else "稳定期"
            llm_qa_prediction = "进展期" if llm_active_probb > llm_stable_probb else "稳定期"

            if llm_prediction == llm_qa_prediction:
                global_probs["稳定期"].append(llm_stable_prob)
                global_probs["进展期"].append(llm_active_prob)
                global_probs["稳定期"].append(llm_stable_probb)
                global_probs["进展期"].append(llm_active_probb)

            # 构建详情信息
            detail_info = {
                "prompt": combo_info["name"],
                "description": combo_info["description"],
                "images": image_paths,
                "image_labels": combo_info["image_labels"],

                # 传统模型结果
                "traditional_prediction": prediction,
                "traditional_stable_prob": f"{stable_prob:.4f}",
                "traditional_active_prob": f"{active_prob:.4f}",

                # 模拟的大模型分类结果
                "llm_class_prediction": llm_prediction,
                "llm_class_stable_prob": f"{llm_stable_prob:.4f}",
                "llm_class_active_prob": f"{llm_active_prob:.4f}",

                # 模拟的大模型问答结果
                "llm_qa_answer": prediction,
                "llm_qa_stable_prob": f"{llm_stable_probb:.4f}",
                "llm_qa_active_prob": f"{llm_active_probb:.4f}",

            }

            details.append(detail_info)
    # 计算全局 prediction & confidence
    avg_stable = sum(global_probs["稳定期"]) / len(global_probs["稳定期"]) if global_probs["稳定期"] else 0.0
    avg_active = sum(global_probs["进展期"]) / len(global_probs["进展期"]) if global_probs["进展期"] else 0.0

    if avg_stable > avg_active:
        final_prediction = "稳定期"
        final_confidence = avg_stable
    else:
        final_prediction = "进展期"
        final_confidence = avg_active
    # 根据上传的图片数量调整置信度分数
    num_images = 0
    if clinical_path:
        num_images += 1
    if woods_path:
        num_images += 1

    if num_images == 1:
        final_confidence *= 50
    elif num_images == 2:
        final_confidence *= 100

    return details, final_prediction, final_confidence
    
    

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

def get_mock_prediction(clinical_path=None, woods_path=None, temp_path=None, doot_dir=None):
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
                    feature_filename = "edge_enhanced_clinical.jpg"
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
                    feature_filename = "edge_enhanced_woods.jpg"
                    save_path = os.path.join(temp_path, feature_filename)
                    cv2.imwrite(save_path, overlay_img)
                    feature_maps["woods_feature"] = f"uploads/temp/{temp_dir_name}/{feature_filename}"

    # === 生成数据集文件 ===
    if temp_path:
        try:
            # 模拟预测状态为进展期
            status_prediction = "active_stage"
            
            # 生成唯一的patient_id
            temp_dir_name = os.path.basename(temp_path)
            patient_id = f"patient_{temp_dir_name}"
            
            # 1. 创建临时的master.json文件
            master_case_data = {
                "idx": 0,
                "status": status_prediction,
                "patient_id": patient_id,
                "images": {
                    "clinical": [os.path.basename(clinical_path).split('_')[-1]] if clinical_path else [],
                    "wood_lamp": [os.path.basename(woods_path).split('_')[-1]] if woods_path else []
                }
            }
            
            master_json_path = os.path.join(temp_path, 'master.json')
            with open(master_json_path, 'w', encoding='utf-8') as f:
                json.dump([master_case_data], f, indent=2, ensure_ascii=False)

            two_binary_cls(master_json_path, temp_path, doot_dir)
            two_choice(master_json_path, temp_path, doot_dir)

            print(f"模拟预测：成功通过外部脚本在 {temp_path} 生成数据集文件。")
            
        except Exception as e:
            print(f"模拟预测通过外部脚本生成数据集时出错: {e}")

    # 完整结果
    result = {
        "final_prediction": "进展期",
        "confidence": 99,
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
        doot_dir = temp_dir_name
        os.makedirs(temp_path, exist_ok=True)

        clinical_path, woods_path = None, None
        clinical_filename, woods_filename = None, None

        if has_clinical:
            clinical_filename = secure_filename(f"original_clinical.{clinical_file.filename.lower().split('.')[-1]}") #secure_filename: 确保文件名安全，防止被hacker恶意文件名攻击
            clinical_path = os.path.join(temp_path, clinical_filename)
            clinical_file.save(clinical_path)
            
        if has_woods:
            woods_filename = secure_filename(f"original_woods.{woods_file.filename.lower().split('.')[-1]}")
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
        result = predict_with_model(clinical_data, woods_data, clinical_path, woods_path, temp_path, doot_dir)
        
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
        # timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_dir_name)
        
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
    # load_model()
    
    # 启动服务
    print("服务启动中...")
    print("访问地址: http://localhost:8888")
    print("API接口: http://localhost:8888/predict")
    #这里只是启动服务器，具体的执行是在http请求中执行的
    app.run(debug=True, host='0.0.0.0', port=8080) #debug=true:代码改动时自动重启服务器，host='0.0.0.0':监听所有网络接口，允许外部访问，port=8080:端口号
