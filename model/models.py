import torch
import torch.nn as nn
from transformers import AutoModel
import os

class SharedBackboneModel:
    """共享主干模型的基类"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def load_trainable_params(self, filepath):
        """加载可训练参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数文件不存在: {filepath}")
        
        trainable_params = torch.load(filepath, map_location='cpu')
        # self.load_state_dict(trainable_params, strict=False)
        if 'fusion' in trainable_params:
            self.fusion.load_state_dict(trainable_params['fusion'])
        if 'classifier' in trainable_params:
            self.classifier.load_state_dict(trainable_params['classifier'])
        print(f"已加载可训练参数从 {filepath}")

class SingleStreamModel(nn.Module, SharedBackboneModel):
    """单流模型（用于oc/ow）"""
    def __init__(self, backbone, num_classes=2):
        nn.Module.__init__(self)
        SharedBackboneModel.__init__(self, backbone)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        features = self.backbone.get_image_features(pixel_values=x)
        return self.classifier(features)

class DualStreamModel(nn.Module, SharedBackboneModel):
    """双流模型（用于oc_ec, ow_ew, oc_ow）"""
    def __init__(self, backbone, num_classes=2):
        nn.Module.__init__(self)
        SharedBackboneModel.__init__(self, backbone)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes))
        
    
    def forward(self, x1, x2):
        features1 = self.backbone.get_image_features(pixel_values=x1)
        features2 = self.backbone.get_image_features(pixel_values=x2)
        
        # 特征融合
        combined = torch.cat((features1, features2), dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

class QuadStreamModel(nn.Module, SharedBackboneModel):
    """四流模型（用于oc_ec_ow_ew）"""
    def __init__(self, backbone, num_classes=2):
        nn.Module.__init__(self)
        SharedBackboneModel.__init__(self, backbone)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(768 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes))
    
    def forward(self, x1, x2, x3, x4):
        features1 = self.backbone.get_image_features(pixel_values=x1)
        features2 = self.backbone.get_image_features(pixel_values=x2)
        features3 = self.backbone.get_image_features(pixel_values=x3)
        features4 = self.backbone.get_image_features(pixel_values=x4)
        
        # 特征融合
        combined = torch.cat((features1, features2, features3, features4), dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)