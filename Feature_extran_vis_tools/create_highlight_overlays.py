# 文件名: Feature_extran_vis_tools/create_highlight_overlays.py

import cv2
import numpy as np

# --- 可视化配置 ---
COLOR_MAPPING = {
    "clinical": cv2.COLORMAP_HOT,
    "wood_lamp": cv2.COLORMAP_COOL
}
ORIGINAL_IMAGE_ALPHA = 0.4
HEATMAP_BETA = 0.6
HIGHLIGHT_THRESHOLD = 5

def create_overlay_image(original_img_path, gradient_map_gray, image_type):
    """
    根据给定的原始图像、梯度图和图像类型，生成高亮叠加图。

    Args:
        original_img_path (str): 原始图像的文件路径。
        gradient_map_gray (np.array): 单通道的灰度梯度图。
        image_type (str): 图像类型 ('clinical' or 'wood_lamp')，用于选择颜色映射。

    Returns:
        np.array: 返回合成后的图像（BGR格式），如果出错则返回 None。
    """
    original_img = cv2.imread(original_img_path)

    if original_img is None:
        print(f"错误: 无法读取原始图像 at {original_img_path}")
        return None
    
    if gradient_map_gray is None:
        print("错误: 传入的梯度图为 None")
        return None

    # --- 使用蒙版技术 ---

    # 1. 创建一个二值蒙版：只有梯度值大于阈值的区域才为白色(255)
    _, mask = cv2.threshold(gradient_map_gray, HIGHLIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 2. 将蒙版转为3通道，以便与彩色图像一起使用
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 3. 确定颜色方案并生成热力图
    colormap = COLOR_MAPPING.get(image_type, cv2.COLORMAP_JET) # 使用传入的 image_type
    heatmap_color = cv2.applyColorMap(gradient_map_gray, colormap)
    
    # 确保热力图和原图尺寸一致
    if original_img.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 4. 将热力图与原图进行半透明融合
    blended_img = cv2.addWeighted(original_img, ORIGINAL_IMAGE_ALPHA, heatmap_color, HEATMAP_BETA, 0)
    
    # 5. 最终合成：使用蒙版决定最终像素
    # 在蒙版为白色的地方，使用融合后的像素；否则，使用原始图像的像素。
    final_img = np.where(mask_3channel > 0, blended_img, original_img)
    
    return final_img
