# 文件名: create_highlight_overlays.py (最终修正版)

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# ========== 1. 配置区域 ==========

# --- 输入路径 ---
ORIGINAL_IMAGES_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/processed")
FEATURE_MAPS_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/edge_features_20250708")

# --- 输出路径 ---
OVERLAY_OUTPUT_ROOT = Path("/hdd/common/datasets/medical-image-analysis/tzb/highlighted_overlays")

# --- 可视化配置 ---
COLOR_MAPPING = {
    "clinical": cv2.COLORMAP_HOT,
    "wood_lamp": cv2.COLORMAP_COOL
}
ORIGINAL_IMAGE_ALPHA = 0.4
HEATMAP_BETA = 0.6

# 新增：设置一个阈值，只有梯度大于这个值的区域才会被高亮
# 这可以过滤掉非常微弱的背景噪声，让高亮区域更干净
HIGHLIGHT_THRESHOLD = 5 


def main():
    """
    主函数，使用蒙版技术执行高亮叠加图的生成任务。
    """
    print("开始生成高亮叠加图...")
    
    if not ORIGINAL_IMAGES_ROOT.exists() or not FEATURE_MAPS_ROOT.exists():
        print("错误: 输入路径不存在，请检查。")
        return

    original_images = list(ORIGINAL_IMAGES_ROOT.rglob("*.jpg"))
    if not original_images:
        print("错误：在原始图像目录中未找到任何 .jpg 文件。")
        return
    
    print(f"找到 {len(original_images)} 张原始图像，开始处理...")
    
    for original_img_path in tqdm(original_images, desc="生成叠加图"):
        try:
            relative_path = original_img_path.relative_to(ORIGINAL_IMAGES_ROOT)
            feature_map_path = (FEATURE_MAPS_ROOT / relative_path).with_suffix('.png')
            overlay_output_path = (OVERLAY_OUTPUT_ROOT / relative_path).with_suffix('.png')
            
            overlay_output_path.parent.mkdir(parents=True, exist_ok=True)

            if feature_map_path.exists():
                original_img = cv2.imread(str(original_img_path))
                gradient_map_gray = cv2.imread(str(feature_map_path), cv2.IMREAD_GRAYSCALE)
                
                if original_img is None or gradient_map_gray is None:
                    shutil.copy2(original_img_path, overlay_output_path)
                    continue
                
                # --- 核心修改：使用蒙版技术 ---
                
                # 1. 创建一个二值蒙版：只有梯度值大于阈值的区域才为白色(255)
                _, mask = cv2.threshold(gradient_map_gray, HIGHLIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
                
                # 2. 将蒙版转为3通道，以便与彩色图像一起使用
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # 3. 确定颜色方案并生成热力图 (同之前)
                path_str = str(original_img_path)
                colormap = COLOR_MAPPING.get('clinical' if 'clinical' in path_str else 'wood_lamp' if 'wood_lamp' in path_str else None, cv2.COLORMAP_JET)
                heatmap_color = cv2.applyColorMap(gradient_map_gray, colormap)
                
                if original_img.shape[:2] != heatmap_color.shape[:2]:
                    heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]))

                # 4. 将热力图与原图进行半透明融合 (同之前)
                blended_img = cv2.addWeighted(original_img, ORIGINAL_IMAGE_ALPHA, heatmap_color, HEATMAP_BETA, 0)
                
                # 5. 最终合成：使用蒙版决定最终像素
                # np.where(condition, value_if_true, value_if_false)
                # 意思是：在蒙版为白色的地方，使用融合后的像素；在蒙版为黑色的地方，使用原始图像的像素。
                final_img = np.where(mask_3channel > 0, blended_img, original_img)
                
                cv2.imwrite(str(overlay_output_path), final_img)

            else:
                # 梯度图不存在，直接复制原图
                shutil.copy2(original_img_path, overlay_output_path)

        except Exception as e:
            print(f"\n处理文件 {original_img_path} 时发生错误: {e}")

    print("\n所有叠加任务完成！")
    print(f"结果已保存至: {OVERLAY_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()