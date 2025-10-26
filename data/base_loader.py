from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
from transformers import Qwen2VLProcessor
import torch

class BaseDataset(Dataset, ABC):
    """基础数据集抽象类"""
    
    def __init__(self, processor=None, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

class BaseDataLoader(ABC):
    """基础数据加载器抽象类"""
    
    def __init__(self, config):
        self.config = config
        self.processor = self._load_processor()
    
    def _load_processor(self):
        """加载Qwen-VL处理器"""
        from transformers import Qwen2VLProcessor
        processor = Qwen2VLProcessor.from_pretrained(self.config.model.base_model)
        return processor
    
    @abstractmethod
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """获取训练、验证、测试数据集 - 子类必须实现"""
        pass
    
    def get_data_loaders(self):
        """获取数据加载器"""
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        ) if val_dataset is not None else None
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        ) if test_dataset is not None else None
        
        return train_loader, val_loader, test_loader