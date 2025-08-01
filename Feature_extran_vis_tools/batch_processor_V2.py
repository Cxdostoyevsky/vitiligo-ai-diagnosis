# 文件名: batch_processor.py (支持回退策略)

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
OUTPUT_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/edge_features_20250708")

# --- 定义您想要尝试的参数配置 ---
PARAMETER_CONFIGS = [
    # --- 组1: 基准线 (作为主要的回退目标) ---

    # --- 组5: 组合策略 (增加回退逻辑) ---
    {
        "name": "config_09_sensitive_and_strong_smoothing",
        # 新增: 如果此配置失败，则回退到'config_01_baseline_c10_k5'
        "fallback_to": "config_01_baseline_c10_k5",
        "params": {
            "adaptive_c": -15,
            "morph_ksize": 7,
            "min_area_ratio": 0.0005,
        }
    },
    {
        "name": "config_01_baseline_c10_k5",
        "params": {
            "adaptive_c": -10, "morph_ksize": 5, "min_area_ratio": 0.0005,
        }
    }
]

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

# --- MODIFIED: process_dataset 函数修改以支持回退 ---
def process_dataset(config, all_configs_map):
    """
    使用单组参数配置来处理整个数据集，并在失败时尝试回退。
    """
    config_name = config["name"]
    params = config["params"]
    
    config_output_dir = OUTPUT_ROOT
    print(f"--- 开始处理配置: {config_name} ---")
    if "fallback_to" in config:
        print(f"    (此配置已启用回退策略，失败后将尝试 '{config['fallback_to']}')")
    print(f"    输出将保存在: {config_output_dir}")

    image_paths = list(SOURCE_ROOT.rglob("*.jpg"))
    if not image_paths:
        print(f"错误：在目录 {SOURCE_ROOT} 中未找到任何 .jpg 文件。")
        return

    all_features = {}
    success_count = 0
    fallback_count = 0

    for img_path in tqdm(image_paths, desc=f"处理 {config_name}"):
        relative_path = img_path.relative_to(SOURCE_ROOT)
        
        # --- 新的回退逻辑 ---
        final_results = None
        succeeded_config_name = None

        # 1. 第一次尝试：使用主配置
        primary_results = extract_woodlamp_edge_features(str(img_path), **params)
        
        if primary_results:
            final_results = primary_results
            succeeded_config_name = config_name
        
        # 2. 如果失败，并且定义了回退策略，则进行第二次尝试
        elif "fallback_to" in config:
            fallback_name = config["fallback_to"]
            if fallback_name in all_configs_map:
                fallback_params = all_configs_map[fallback_name]
                fallback_results = extract_woodlamp_edge_features(str(img_path), **fallback_params)
                
                if fallback_results:
                    final_results = fallback_results
                    succeeded_config_name = fallback_name # 记录成功的是回退配置
                    fallback_count += 1 # 统计回退次数
            else:
                # 这种情况一般不会发生，除非配置名写错了
                print(f"\n警告: 未找到定义的回退配置'{fallback_name}'")
        
        # --- 结束回退逻辑 ---

        # 如果最终有结果 (无论是主配置还是回退配置)
        if final_results:
            success_count += 1
            # 关键：在特征中记录下是哪个配置最终成功了
            final_results["features"]["succeeded_with_config"] = succeeded_config_name
            
            # 保存梯度图
            gradient_map_output_path = (config_output_dir / relative_path).with_suffix('.png')
            gradient_map_output_path.parent.mkdir(parents=True, exist_ok=True)
            save_gradient_map(final_results["gradient_map"], gradient_map_output_path)
            
            # 存储特征指标
            all_features[str(relative_path)] = final_results["features"]

    # ... (JSON保存部分略作修改以显示统计信息) ...
    if all_features:
        json_output_path = config_output_dir / "features.json"
        print(f"处理完成。共成功处理 {success_count} / {len(image_paths)} 张图片。")
        if fallback_count > 0:
            print(f"    其中 {fallback_count} 张图片使用了回退策略。")
        print(f"正在将特征记录保存到 {json_output_path}...")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, indent=4, ensure_ascii=False)
        print("保存成功。")
    else:
        print("警告：没有成功处理任何图像，未生成 features.json 文件。")
    
    print(f"--- 配置 {config_name} 处理结束 ---\n")


# --- MODIFIED: 主函数修改以支持回退 ---
if __name__ == "__main__":
    if not SOURCE_ROOT.exists():
        print(f"错误：源目录不存在，请检查路径: {SOURCE_ROOT}")
    else:
        # NEW: 创建一个从配置名称到参数的映射，方便快速查找回退配置
        configs_by_name = {c['name']: c['params'] for c in PARAMETER_CONFIGS}
        
        for configuration in PARAMETER_CONFIGS:
            # 将配置本身和配置映射表都传给处理函数
            process_dataset(configuration, configs_by_name)
        print("所有任务已完成！")