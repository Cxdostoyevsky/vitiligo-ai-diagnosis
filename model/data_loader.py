import json
import os
import torch
from PIL import Image
from config import IMAGE_ROOT_DIR, INPUT_CONFIG

def extract_id_from_path(image_path):
    """从图像路径中提取ID"""
    parts = image_path.split('/')
    folder_id = parts[0]
    file_id = parts[-1].split('.')[0]  
    return f"{folder_id}_{0}"

def load_single_image(image_path, processor):
    """加载并处理单张图像"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像不存在: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].squeeze(0)

def load_test_data(processor=None, input_config=None, image_root_dir=None):
    """加载测试数据 - 使用外部传入的处理器"""
    
    # 使用传入的配置，如果没有则使用默认配置
    if input_config is None:
        input_config = INPUT_CONFIG
    if image_root_dir is None:
        image_root_dir = IMAGE_ROOT_DIR
    
    # 检查input_config是否为空
    if not input_config:
        raise ValueError("input_config不能为空，至少需要一个输入类型配置")
    
    # 第一步：确定主顺序（使用第一个输入类型作为参考）
    primary_type = next(iter(input_config))
    primary_json = input_config[primary_type]["json_path"]
    # 读取主JSON并记录ID顺序
    with open(primary_json, 'r', encoding='utf-8') as f:
        primary_data = json.load(f)   
    # 提取主顺序的样本ID列表
    ordered_ids = []
    for item in primary_data:
        sample_id = extract_id_from_path(item["image"][0])
        ordered_ids.append(sample_id)   
    
    # 第二步：创建有序样本字典
    test_samples = {}
    for sample_id in ordered_ids:
        test_samples[sample_id] = {
            "id": sample_id,
            "images": {}
        }
    
    # 第三步：按输入类型加载数据，保持主顺序
    for input_type, config in input_config.items():
        with open(config["json_path"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建当前JSON的ID到数据的映射
        id_to_data = {}
        for item in data:
            item_id = extract_id_from_path(item["image"][0])
            id_to_data[item_id] = item
        
        # 按照主顺序处理样本
        for sample_id in ordered_ids:
            if sample_id in id_to_data:
                item = id_to_data[sample_id]
            else:
                print(f"警告: 样本 {sample_id} 在 {input_type} JSON中缺失")
                item = {"image": []}  # 创建空条目
            
            images = []
            for img_path in item["image"]:
                full_path = os.path.join(image_root_dir, img_path)
                try:
                    img_tensor = load_single_image(full_path, processor)
                    images.append(img_tensor)
                except FileNotFoundError:
                    placeholder = torch.zeros((3, 512, 512))
                    images.append(placeholder)
                    print(f"警告: {input_type} 图像未找到: {img_path}")
            
            test_samples[sample_id]["images"][input_type] = images
    
    # # 第四步：验证完整性
    # missing_in_types = {}
    # for sample_id in ordered_ids:
    #     for input_type in INPUT_CONFIG:
    #         if input_type not in test_samples[sample_id]["images"]:
    #             if input_type not in missing_in_types:
    #                 missing_in_types[input_type] = []
    #             missing_in_types[input_type].append(sample_id)
    
    # for input_type, samples in missing_in_types.items():
    #     print(f"警告: {len(samples)} 个样本在 {input_type} 中缺失数据")
    #     # 添加占位符
    #     num_images = len(INPUT_CONFIG[input_type]["image_types"])
    #     for sample_id in samples:
    #         test_samples[sample_id]["images"][input_type] = [
    #             torch.zeros((3, 512, 512)) for _ in range(num_images)]
            
    return ordered_ids, test_samples