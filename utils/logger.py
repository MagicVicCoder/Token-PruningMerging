import logging
import os
import sys
from datetime import datetime
from typing import Optional
import torch

def setup_logger(output_dir: str, name: str = "qwen_pruning") -> logging.Logger:
    """设置日志记录器"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件处理器
    log_file = os.path.join(output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, output_dir: str):
        self.logger = setup_logger(output_dir)
        self.output_dir = output_dir
        
        # TensorBoard支持
        self.tensorboard_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(output_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_dir)
        except ImportError:
            self.logger.warning("TensorBoard not available")
    
    def log_metrics(self, step: int, metrics: dict, prefix: str = ""):
        """记录指标"""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {prefix} - {metrics_str}")
        
        # TensorBoard记录
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{prefix}/{key}", value, step)
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def log_model_stats(self, step: int, stats: dict):
        """记录模型统计信息"""
        stats_str = " - ".join([f"{k}: {v:.4f}" for k, v in stats.items()])
        self.logger.info(f"Step {step} - Model Stats - {stats_str}")
        
        if self.tensorboard_writer:
            for key, value in stats.items():
                self.tensorboard_writer.add_scalar(f"model/{key}", value, step)
    
    def close(self):
        """关闭记录器"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()