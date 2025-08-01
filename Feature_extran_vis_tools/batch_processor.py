# 文件名: batch_processor.py

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 从我们自己的模块中导入核心处理函数
from feature_extractor import extract_woodlamp_edge_features

# ========== 1. 配置区域 ==========

# 定义输入和输出的根目录
SOURCE_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/processed")
OUTPUT_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/edge_features_20250705")

# --- 定义您想要尝试的参数配置 ---
# 这是一个列表，您可以添加任意多组配置。
# 'name': 将用于创建文件夹名，请使用合法的文件名字符。
# 'params': 这是一个字典，键必须与 extract_woodlamp_edge_features 函数的参数名完全匹配。
PARAMETER_CONFIGS = [
    # --- 组1: 基准线 ---
    {
        "name": "config_01_baseline_c10_k5",
        "params": {
            "adaptive_c": -10,
            "morph_ksize": 5,
            "min_area_ratio": 0.0005,
        }
    },
    
    # --- 组2: 探索分割灵敏度 (adaptive_c) ---
    {
        "name": "config_02_less_sensitive_c5",
        "params": {
            "adaptive_c": -5,  # 灵敏度降低，适合边界非常清晰、对比度高的图像
            "morph_ksize": 5,
            "min_area_ratio": 0.0005,
        }
    },
    {
        "name": "config_03_more_sensitive_c15",
        "params": {
            "adaptive_c": -15, # 灵敏度提升 (原配置之一)
            "morph_ksize": 5,
            "min_area_ratio": 0.0005,
        }
    },
    {
        "name": "config_04_very_sensitive_c20",
        "params": {
            "adaptive_c": -20, # 灵敏度极高，尝试捕捉最细微的边界，但噪声风险也最大
            "morph_ksize": 5,
            "min_area_ratio": 0.0005,
        }
    },

    # --- 组3: 探索平滑与去噪强度 (morph_ksize) ---
    {
        "name": "config_05_weak_smoothing_k3",
        "params": {
            "adaptive_c": -10,
            "morph_ksize": 3,  # 平滑/去噪较弱，保留更多轮廓细节，但也可能保留更多小噪点
            "min_area_ratio": 0.0005,
        }
    },
    {
        "name": "config_06_strong_smoothing_k7",
        "params": {
            "adaptive_c": -10,
            "morph_ksize": 7,  # 平滑/去噪更强 (原配置之一)
            "min_area_ratio": 0.0005,
        }
    },

    # --- 组4: 探索面积过滤阈值 (min_area_ratio) ---
    {
        "name": "config_07_strict_area_filter",
        "params": {
            "adaptive_c": -10,
            "morph_ksize": 5,
            "min_area_ratio": 0.001, # 更严格，只保留面积大于图像0.1%的区域，强力过滤小斑点
        }
    },
    {
        "name": "config_08_loose_area_filter",
        "params": {
            "adaptive_c": -10,
            "morph_ksize": 5,
            "min_area_ratio": 0.0001, # 更宽松，会保留非常小的区域，适合检测早期微小皮损
        }
    },
    
    # --- 组5: 组合策略 ---
    {
        "name": "config_09_sensitive_and_strong_smoothing",
        "params": {
            "adaptive_c": -15, # 使用高灵敏度检测
            "morph_ksize": 7,   # 同时使用强平滑来抑制高灵敏度带来的噪声
            "min_area_ratio": 0.0005,
        }
    }
]
    # 您可以在这里添加更多配置...
    # {
    #     "name": "config_name_4",
    #     "params": { ... }
    # }


def save_gradient_map(gradient_map, output_path):
    """将浮点型的梯度幅度图转换为可保存的8位图像"""
    # 归一化到 0-255 范围
    normalized_map = cv2.normalize(gradient_map, None, 0, 255, cv2.NORM_MINMAX)
    uint8_map = normalized_map.astype(np.uint8)
    
    # 如果希望保存为彩色的热力图，可以取消下面的注释
    # colored_map = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
    # cv2.imwrite(str(output_path), colored_map)

    # 保存为灰度图
    cv2.imwrite(str(output_path), uint8_map)


def process_dataset(config):
    """
    使用单组参数配置来处理整个数据集。
    """
    config_name = config["name"]
    params = config["params"]
    
    # 构建此配置的输出目录
    config_output_dir = OUTPUT_ROOT / config_name
    print(f"--- 开始处理配置: {config_name} ---")
    print(f"输出将保存在: {config_output_dir}")

    # 查找所有jpg图像
    image_paths = list(SOURCE_ROOT.rglob("*.jpg"))
    if not image_paths:
        print(f"错误：在目录 {SOURCE_ROOT} 中未找到任何 .jpg 文件。")
        return

    # 用于存储所有图片特征的字典
    all_features = {}

    # 使用tqdm创建进度条
    for img_path in tqdm(image_paths, desc=f"处理 {config_name}"):
        # 计算相对路径，用于构建输出路径和作为JSON的键
        relative_path = img_path.relative_to(SOURCE_ROOT)
        
        # 调用核心函数进行处理
        results = extract_woodlamp_edge_features(str(img_path), **params)

        # 如果处理成功
        if results:
            # 1. 保存梯度幅度图
            gradient_map_output_path = config_output_dir / relative_path
            # 创建父目录（如果不存在）
            gradient_map_output_path.parent.mkdir(parents=True, exist_ok=True)
            save_gradient_map(results["gradient_map"], gradient_map_output_path)

            # 2. 存储特征指标
            # 使用 str(relative_path) 作为key，确保是可序列化的字符串
            all_features[str(relative_path)] = results["features"]

    # 所有图片处理完毕后，保存特征指标为JSON文件
    if all_features:
        json_output_path = config_output_dir / "features.json"
        print(f"处理完成。正在将 {len(all_features)} 条特征记录保存到 {json_output_path}...")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=4, ensure_ascii=False)
        print("保存成功。")
    else:
        print("警告：没有成功处理任何图像，未生成 features.json 文件。")
    
    print(f"--- 配置 {config_name} 处理结束 ---\n")


if __name__ == "__main__":
    # 检查输出根目录是否存在
    if not SOURCE_ROOT.exists():
        print(f"错误：源目录不存在，请检查路径: {SOURCE_ROOT}")
    else:
        # 遍历所有参数配置并执行处理
        for configuration in PARAMETER_CONFIGS:
            process_dataset(configuration)
        print("所有任务已完成！")