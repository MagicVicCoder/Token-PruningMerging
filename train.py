import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict
from typing import Dict, List, Tuple, Optional, Any

from config import ConfigManager
from data import get_data_loader_class
from model.pruned_model import QwenVLPrunedModel
from evaluator.evaluator import ModelEvaluator
from utils.logger import TrainingLogger
from utils.downloader import DatasetManager

class Trainer:
    """模型训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # 初始化组件
        self.setup_components()
    
    def setup_components(self):
        """初始化训练组件"""
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 初始化日志记录器
        self.logger = TrainingLogger(self.config.output_dir)
        
        # 确保数据和模型已准备
        if not DatasetManager.ensure_dataset_ready(self.config):
            raise RuntimeError("Dataset is not ready. Please check the dataset path.")
        
        # 加载数据
        self.logger.log_info(f"Loading {self.config.data.dataset_name} data...")
        try:
            data_loader_class = get_data_loader_class(self.config.data.dataset_name)
            data_loader = data_loader_class(self.config)
            self.train_loader, self.val_loader, self.test_loader = data_loader.get_data_loaders()
        except Exception as e:
            self.logger.log_error(f"Failed to load data: {e}")
            raise
        
        # 加载模型
        self.logger.log_info("Loading model...")
        try:
            self.model = QwenVLPrunedModel(self.config, self.config.model)
            self.model.to(self.device)
        except Exception as e:
            self.logger.log_error(f"Failed to load model: {e}")
            raise
        
        # 初始化优化器和调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # 初始化评估器
        self.evaluator = ModelEvaluator(self.model, self.device, self.logger)
        
        self.logger.log_info(f"Training setup completed on device: {self.device}")
        self.logger.log_info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.log_info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.log_info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch.get('attention_mask', None),
                labels=batch.get('labels', None)
            )
            
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # 定期评估和记录
            if (step + 1) % self.config.training.eval_steps == 0:
                # 记录训练指标
                train_metrics = {
                    'loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                self.logger.log_metrics(epoch * len(self.train_loader) + step, train_metrics, "train")
                
                # 记录模型统计信息
                model_stats = self.model.get_pruning_statistics()
                if model_stats:
                    self.logger.log_model_stats(epoch * len(self.train_loader) + step, model_stats)
            
            # 定期保存检查点
            if (step + 1) % self.config.training.save_steps == 0:
                self.save_checkpoint(epoch, step)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.logger.log_info(f"Validating after epoch {epoch+1}...")
        val_metrics = self.evaluator.evaluate(self.val_loader, f"Epoch {epoch+1} Validation")
        return val_metrics
    
    def save_checkpoint(self, epoch: int, step: int):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"checkpoint_epoch_{epoch+1}_step_{step+1}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.log_info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """训练主循环"""
        self.logger.log_info("Starting training...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            val_loss = val_metrics.get('avg_loss', float('inf'))
            
            # 记录epoch指标
            epoch_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': time.time() - epoch_start_time
            }
            self.logger.log_metrics((epoch + 1) * len(self.train_loader), epoch_metrics, "epoch")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.log_info(f"Best model saved with val_loss: {val_loss:.4f}")
        
        total_time = time.time() - start_time
        self.logger.log_info(f"Training completed in {total_time:.2f} seconds")
        
        # 最终测试
        self.final_test()
        
        self.logger.close()
    
    def final_test(self):
        """最终测试"""
        self.logger.log_info("Running final test...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.config.output_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.log_info("Loaded best model for testing")
        
        test_metrics = self.evaluator.evaluate(self.test_loader, "Final Test")
        
        # 保存测试结果
        results_path = os.path.join(self.config.output_dir, "test_results.txt")
        with open(results_path, 'w') as f:
            f.write("Test Results:\n")
            for key, value in test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        self.logger.log_info(f"Test results saved to: {results_path}")

def main():
    """主训练函数"""
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get_experiment_config()
    
    # 训练模型
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
