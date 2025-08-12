# 配置参数
import os

# CUDA_DEVICE
DEVICE = 2

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径配置
MODEL_PATHS = {
    "oc": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OC/5/final_trainable_params_OC.pth"),
    "ow": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OW/5/final_trainable_params_OW.pth"),
    "oc_ec": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OC_EC/5/final_trainable_params_OC_EC.pth"),
    "ow_ew": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OW_EW/5/final_trainable_params_OC_EC.pth"),
    "oc_ow": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OC_OW/5/final_trainable_params_OC_OW.pth"),
    "oc_ec_ow_ew": os.path.join(BASE_DIR, "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2_OC_OW_EC_EW/5/final_trainable_params_OC_OW_EC_EW.pth"),
    # "oc_ec_ow_ew": os.path.join(BASE_DIR, "models/oc_ec_ow_ew_trainable_params.pth"),
}

# SigLIP 主干模型路径
SIGLIP_MODEL_PATH = "/hdd/common/datasets/medical-image-analysis/tzb/siglip2/siglip2-base-patch16-512"
# 处理器路径（与模型路径相同）
SIGLIP_PROCESSOR_PATH = SIGLIP_MODEL_PATH

# 图像根目录
IMAGE_ROOT_DIR = "/hdd/chenxi/bzt/uploads_images"

# 输出文件路径
PROBABILITIES_CSV = os.path.join(BASE_DIR, "results/probabilities_tt.csv")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "results/predictions_tt.csv")

# 输入类型映射和对应的JSON路径
INPUT_CONFIG = {
    "oc": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_1img_OC.json",
        "image_types": ["clinical"]
    },
    "ow": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_1img_OW.json",
        "image_types": ["wood"]
    },
    "oc_ec": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OC_EC.json",
        "image_types": ["clinical", "edge_enhanced_clinical"]
    },
    "ow_ew": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OW_EW.json",
        "image_types": ["wood", "edge_enhanced_wood"]
    },
    "oc_ow": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_2img_OC_OW.json",
        "image_types": ["clinical", "wood"]
    },
    "oc_ec_ow_ew": {
        "json_path": "/hdd/chenxi/bzt/uploads_images/20250809_104721_546043/train_binary_cls_4img_OC_EC_OW_EW.json",
        "image_types": ["clinical", "edge_enhanced_clinical", "wood", "edge_enhanced_wood"]
    },
}