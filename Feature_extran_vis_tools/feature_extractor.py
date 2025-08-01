# 文件名: feature_extractor.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_woodlamp_edge_features(image_path, **kwargs):
    """
    这是我们之前最终确定的特征提取函数。
    它接受一个 image_path 和一系列参数作为输入，
    返回一个包含特征和可视化图像的字典。
    (代码和上一轮的最终版完全一样，这里为了简洁省略，
    请确保您将上一轮的完整函数复制到这里)
    """
    # ========== 1. 图像加载与预处理 ==========
    img = cv2.imread(image_path)
    if img is None:
        # 在批处理中，我们不抛出异常，而是返回None让主程序处理
        print(f"警告: 无法读取图像: {image_path}")
        return None

    # 从kwargs中获取参数，如果未提供则使用默认值
    enhance_alpha = kwargs.get('enhance_alpha', 1.5)
    enhance_beta = kwargs.get('enhance_beta', -0.5)
    overexpose_thresh = kwargs.get('overexpose_thresh', 220)
    adaptive_c = kwargs.get('adaptive_c', -10)
    morph_ksize = kwargs.get('morph_ksize', 5)
    open_iter = kwargs.get('open_iter', 1)
    close_iter = kwargs.get('close_iter', 2)
    min_area_ratio = kwargs.get('min_area_ratio', 0.0005)
    canny_thresh1 = kwargs.get('canny_thresh1', 100)
    canny_thresh2 = kwargs.get('canny_thresh2', 200)

    # ... (HSV, 增强, 分割, 形态学, 轮廓提取等)
    # ...
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ksize = max(5, min(img.shape[0] // 20, img.shape[1] // 20))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(v, (ksize, ksize), 0)
    v_enhanced = cv2.addWeighted(v, enhance_alpha, blurred, enhance_beta, 0)
    _, overexposed = cv2.threshold(v_enhanced, overexpose_thresh, 255, cv2.THRESH_BINARY)
    v_enhanced = np.where(overexposed > 0, v, v_enhanced)
    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    block_size = min(201, max(51, min(img.shape[0], img.shape[1]) // 5))
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c)
    kernel = np.ones((morph_ksize, morph_ksize), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    min_area = img_area * min_area_ratio
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not large_contours:
        # print(f"警告: 在图像 {image_path} 中未找到有效皮损区域。")
        return None

    # ========== 4. 边界模糊度量化 ==========
    gray_original_blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    grad_x = cv2.Sobel(gray_original_blur, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_original_blur, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = cv2.Canny(closed, canny_thresh1, canny_thresh2)
    edge_gradients = gradient_magnitude[edge_mask > 0]
    boundary_metrics = {}
    if len(edge_gradients) > 0:
        boundary_metrics["mean_gradient"] = np.mean(edge_gradients)
        boundary_metrics["std_gradient"] = np.std(edge_gradients)
        boundary_metrics["max_gradient"] = np.max(edge_gradients)
    else:
        boundary_metrics["mean_gradient"] = 0
        boundary_metrics["std_gradient"] = 0
        boundary_metrics["max_gradient"] = 0
    boundary_metrics["boundary_area_ratio"] = cv2.countNonZero(edge_mask) / img_area
    
    # 返回需要的数据
    return {
        "features": boundary_metrics,
        "gradient_map": gradient_magnitude
    }