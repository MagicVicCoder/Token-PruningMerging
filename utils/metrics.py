import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Any, Optional

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """计算准确率"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, predictions)

def compute_f1(predictions: torch.Tensor, labels: torch.Tensor, average: str = 'macro') -> float:
    """计算F1分数"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, predictions, average=average)

def compute_precision_recall(predictions: torch.Tensor, labels: torch.Tensor, average: str = 'macro') -> Tuple[float, float]:
    """计算精确率和召回率"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, predictions, average=average)
    recall = recall_score(labels, predictions, average=average)
    return precision, recall

def compute_perplexity(loss: float) -> float:
    """根据损失计算困惑度"""
    return float(np.exp(loss))

class MetricCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置状态"""
        self.losses = []
        self.predictions = []
        self.labels = []
        self.logits = []
    
    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        """更新状态"""
        self.losses.append(loss)
        
        # 获取预测结果
        preds = torch.argmax(logits, dim=-1)
        self.predictions.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.logits.append(logits.cpu().detach().numpy())
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        if not self.losses:
            return {}
        
        avg_loss = np.mean(self.losses)
        accuracy = compute_accuracy(torch.tensor(self.predictions), torch.tensor(self.labels))
        f1 = compute_f1(torch.tensor(self.predictions), torch.tensor(self.labels))
        precision, recall = compute_precision_recall(torch.tensor(self.predictions), torch.tensor(self.labels))
        perplexity = compute_perplexity(avg_loss)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'perplexity': perplexity
        }
        
        return metrics
    
    def compute_detailed_metrics(self) -> Dict[str, Any]:
        """计算详细指标"""
        base_metrics = self.compute()
        
        if not base_metrics:
            return {}
        
        # 添加额外统计信息
        detailed_metrics = base_metrics.copy()
        detailed_metrics['total_samples'] = len(self.predictions)
        detailed_metrics['class_distribution'] = np.bincount(self.labels).tolist()
        
        return detailed_metrics