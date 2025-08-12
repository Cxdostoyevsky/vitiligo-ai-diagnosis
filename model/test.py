import os
import torch
import pandas as pd
import time
from tqdm import tqdm
from data_loader import load_test_data
from transformers import AutoProcessor, AutoModel
from models import SingleStreamModel, DualStreamModel, QuadStreamModel
from config import (DEVICE, MODEL_PATHS, SIGLIP_MODEL_PATH, SIGLIP_PROCESSOR_PATH, PROBABILITIES_CSV, PREDICTIONS_CSV, INPUT_CONFIG)

def format_time(seconds):
    """格式化时间输出"""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)} min {secs:.2f} s"


class InferenceSystem:
    """推理系统 - 提前加载所有资源"""
    def __init__(self, device):
        self.device = device
        self.processor = None
        self.backbone = None
        self.models = {}
        
    def initialize(self):
        """初始化系统 - 加载处理器、主干和所有模型"""
        # 加载处理器
        print("加载SigLIP处理器...")
        start = time.time()
        self.processor = AutoProcessor.from_pretrained(SIGLIP_PROCESSOR_PATH)
        print(f"处理器加载完成，耗时: {format_time(time.time() - start)}")
        
        # 加载主干模型
        print("加载SigLIP主干模型...")
        start = time.time()
        self.backbone = AutoModel.from_pretrained(SIGLIP_MODEL_PATH).to(self.device)
        self.backbone.eval()
        print(f"主干模型加载完成，耗时: {format_time(time.time() - start)}")
        
        # 加载所有分类头
        print("加载分类头模型...")
        start = time.time()
        self.models = {
            "oc": SingleStreamModel(self.backbone).to(self.device),
            "ow": SingleStreamModel(self.backbone).to(self.device),
            "oc_ec": DualStreamModel(self.backbone).to(self.device),
            "ow_ew": DualStreamModel(self.backbone).to(self.device),
            "oc_ow": DualStreamModel(self.backbone).to(self.device),
            "oc_ec_ow_ew": QuadStreamModel(self.backbone).to(self.device),
        }
        
        # 加载训练好的参数
        for input_type, model in self.models.items():
            model_path = MODEL_PATHS[input_type]
            model.load_trainable_params(model_path)
            # for name, param in model.named_parameters():
            #     if name.startswith('classifier'):
            #         print(f"{name}: {param}")
            model.eval()
        
        print(f"所有分类头加载完成，耗时: {format_time(time.time() - start)}")
 
    
    def load_data(self, input_config=None, image_root_dir=None):
        """加载测试数据"""
        print("加载测试数据...")
        start = time.time()
        # 如果没有传入配置，使用默认配置
        if input_config is None:
            from config import INPUT_CONFIG
            input_config = INPUT_CONFIG
        ordered_ids, test_samples = load_test_data(processor=self.processor, input_config=input_config, image_root_dir=image_root_dir)
        print(f"加载 {len(test_samples)} 个测试样本，耗时: {format_time(time.time() - start)}")
        return ordered_ids, test_samples
    
    def run_inference(self, ordered_ids, test_samples):
        """运行推理"""
        print("开始推理...")
        start_time = time.time()
        
        probabilities_data = []
        predictions_data = []
        
        progress_bar = tqdm(ordered_ids, desc="推理进度", unit="样本")
        
        for sample_id in progress_bar:
            sample_data = test_samples[sample_id]
            prob_row = {"id": sample_id}
            pred_row = {"id": sample_id}
            
            for input_type, model in self.models.items():
                if input_type not in sample_data["images"]:
                    print(f"错误: 样本 {sample_id} 缺少 {input_type} 图像数据")
                    continue
                    
                images = sample_data["images"][input_type]
                input_tensors = [img.unsqueeze(0).to(self.device) for img in images]
                
                with torch.no_grad():
                    if input_type in ["oc", "ow"]:
                        outputs = model(*input_tensors)
                    elif input_type in ["oc_ec", "ow_ew", "oc_ow"]:
                        outputs = model(*input_tensors[:2])
                    else:  # oc_ec_ow_ew
                        outputs = model(*input_tensors)
                
                probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
                prob_row[f"{input_type}_stable"] = probs[0]
                prob_row[f"{input_type}_active"] = probs[1]
                pred_row[input_type] = torch.argmax(outputs).item()
            
            probabilities_data.append(prob_row)
            predictions_data.append(pred_row)
            progress_bar.update(1)
        
        progress_bar.close()
        
        total_time = time.time() - start_time
        print(f"推理完成，共处理 {len(test_samples)} 个样本，耗时: {format_time(total_time)}")
        # print(f"样本处理速度: {len(test_samples)/total_time:.2f} 样本/秒")
        
        return probabilities_data, predictions_data

def save_results(probabilities_data, predictions_data):
    """保存结果到CSV文件并返回增强的概率数据"""
    print("保存结果...")
    prob_df = pd.DataFrame(probabilities_data)
    pred_df = pd.DataFrame(predictions_data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(PROBABILITIES_CSV), exist_ok=True)
    
    # 保存CSV
    prob_df.to_csv(PROBABILITIES_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    
    print(f"概率结果已保存到: {PROBABILITIES_CSV}")
    print(f"预测结果已保存到: {PREDICTIONS_CSV}")
    
    stable_cols = [c for c in prob_df.columns if c.endswith('_stable')]
    active_cols = [c for c in prob_df.columns if c.endswith('_active')]

    prob_df['sum_stable'] = prob_df[stable_cols].sum(axis=1)
    prob_df['sum_active'] = prob_df[active_cols].sum(axis=1)

    prob_df['result'] = (prob_df['sum_active'] > prob_df['sum_stable']).astype(int)

    sums = prob_df[['sum_stable', 'sum_active']].values
    prob_df['confidence'] = sums.max(axis=1) / sums.sum(axis=1)

    # df.drop(columns=['sum_stable', 'sum_active'], inplace=True)

    prob_df.to_csv('/hdd/chenxi/bzt/model/test_code_zx/results/probabilities_vote.csv', index=False)
    
    # 将增强后的数据转回字典列表格式，供前端使用
    enhanced_probabilities_data = prob_df.to_dict('records')
    return enhanced_probabilities_data

def main():
    # 设置设备
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化推理系统
    inference_system = InferenceSystem(device)
    inference_system.initialize()
    
    # 加载数据
    ordered_ids, test_samples = inference_system.load_data()
    print(f"加载了 {len(ordered_ids)} 个样本ID，样本数量: {len(test_samples)}")
    print(ordered_ids)
    print(test_samples.keys())
    
    # 运行推理
    probabilities_data, predictions_data = inference_system.run_inference(ordered_ids, test_samples)

    # 保存结果
    save_results(probabilities_data, predictions_data)
    print("测试完成!")

if __name__ == "__main__":
    main()