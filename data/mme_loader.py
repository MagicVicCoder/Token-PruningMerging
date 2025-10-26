import os
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .base_loader import BaseDataset, BaseDataLoader

class MMEDataset(BaseDataset):
    """MME数据集 - 使用Hugging Face datasets加载"""
    
    def __init__(self, dataset_split, processor=None, max_length: int = 512):
        self.dataset_split = dataset_split
        self.processor = processor
        self.max_length = max_length
        # 不再需要手动加载样本，直接使用dataset_split
    
    def __len__(self):
        return len(self.dataset_split)
    
    def __getitem__(self, idx):
        sample = self.dataset_split[idx]
        
        # 处理图像
        image = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB') if isinstance(image, str) else image
        
        # 处理文本 - 根据MME数据集的实际结构调整
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        question_id = sample.get('question_id', str(idx))
        
        # 构造输入文本
        text = f"Question: {question}\nAnswer: {answer}" if answer else f"Question: {question}"
        
        if self.processor:
            inputs = self.processor(
                text=text,
                images=image,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # 移除batch维度
            for key in inputs:
                inputs[key] = inputs[key].squeeze(0)
            
            # 添加标签（如果有）
            if answer:
                # 对于分类任务，你可能需要将答案转换为标签ID
                # 这里暂时使用简单的二分类示例
                inputs['labels'] = torch.tensor([1] if answer else [0])
            
            inputs['question_id'] = question_id
        else:
            inputs = {
                'image': image,
                'text': text,
                'question_id': question_id
            }
            if answer:
                inputs['answer'] = answer
        
        return inputs

class MMEDataLoader(BaseDataLoader):
    """MME数据加载器 - 使用Hugging Face datasets"""
    
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """获取训练、验证、测试数据集"""
        data_config = self.config.data
        
        # 从Hugging Face加载数据集
        print("Loading MME dataset from Hugging Face...")
        try:
            dataset = load_dataset("lmms-lab/MME")
            print(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # 根据数据集的实际情况调整split名称
        # 假设数据集有'train', 'validation', 'test'分割
        split_mapping = {
            'train': 'train',
            'val': 'validation', 
            'test': 'test'
        }
        
        # 如果数据集没有标准分割，我们手动划分
        if 'train' not in dataset:
            print("Dataset doesn't have standard splits, creating custom splits...")
            # 这里可以根据需要实现自定义划分逻辑
            # 暂时使用第一个可用的split
            available_split = list(dataset.keys())[0]
            full_dataset = dataset[available_split]
            
            # 简单的划分逻辑
            dataset_size = len(full_dataset)
            train_size = int(dataset_size * data_config.train_split)
            val_size = int(dataset_size * data_config.val_split)
            
            # 使用select方法创建分割
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, dataset_size))
            
            train_dataset = full_dataset.select(train_indices)
            val_dataset = full_dataset.select(val_indices)
            test_dataset = full_dataset.select(test_indices)
        else:
            # 使用标准分割
            train_dataset = dataset[split_mapping['train']]
            val_dataset = dataset[split_mapping['val']] if split_mapping['val'] in dataset else None
            test_dataset = dataset[split_mapping['test']] if split_mapping['test'] in dataset else None
            
            # 如果没有验证集，从训练集划分
            if val_dataset is None:
                train_val_split = train_dataset.train_test_split(
                    test_size=data_config.val_split,
                    seed=self.config.seed
                )
                train_dataset = train_val_split['train']
                val_dataset = train_val_split['test']
        
        # 创建数据集实例
        train_mme_dataset = MMEDataset(
            dataset_split=train_dataset,
            processor=self.processor,
            max_length=data_config.max_length
        )
        
        val_mme_dataset = MMEDataset(
            dataset_split=val_dataset,
            processor=self.processor,
            max_length=data_config.max_length
        ) if val_dataset is not None else None
        
        test_mme_dataset = MMEDataset(
            dataset_split=test_dataset,
            processor=self.processor,
            max_length=data_config.max_length
        ) if test_dataset is not None else None
        
        # 如果没有测试集，使用验证集
        if test_mme_dataset is None and val_mme_dataset is not None:
            test_mme_dataset = val_mme_dataset
        
        return train_mme_dataset, val_mme_dataset, test_mme_dataset
    
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