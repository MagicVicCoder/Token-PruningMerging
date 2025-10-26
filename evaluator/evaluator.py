import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import numpy as np

from utils.metrics import MetricCalculator

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device, logger):
        self.model = model
        self.device = device
        self.logger = logger
        self.metric_calculator = MetricCalculator()
    
    def evaluate(self, dataloader, description: str = "Evaluation") -> Dict[str, float]:
        """在给定数据加载器上评估模型"""
        self.model.eval()
        self.metric_calculator.reset()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=description):
                # 移动数据到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['pixel_values'],
                    attention_mask=batch.get('attention_mask', None),
                    labels=batch.get('labels', None)
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else None
                logits = outputs.logits
                
                # 更新指标
                if loss is not None:
                    total_loss += loss.item() * batch['input_ids'].size(0)
                    total_samples += batch['input_ids'].size(0)
                    
                    self.metric_calculator.update(
                        loss.item(), 
                        logits, 
                        batch.get('labels', torch.zeros_like(batch['input_ids'][:, 0]))
                    )
        
        # 计算指标
        metrics = self.metric_calculator.compute()
        
        if total_samples > 0:
            metrics['avg_loss'] = total_loss / total_samples
        
        # 获取剪枝统计信息
        pruning_stats = self.model.get_pruning_statistics()
        metrics.update(pruning_stats)
        
        self.logger.log_info(f"{description} Results:")
        for key, value in metrics.items():
            self.logger.log_info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def predict(self, dataloader) -> Tuple[List, List]:
        """生成预测"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['pixel_values'],
                    attention_mask=batch.get('attention_mask', None)
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch.get('labels', torch.zeros_like(predictions)).cpu().numpy())
        
        return all_predictions, all_labels