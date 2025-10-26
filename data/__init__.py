from .mme_loader import MMEDataset, MMEDataLoader
from .base_loader import BaseDataset, BaseDataLoader

# 数据集注册表
DATASET_REGISTRY = {
    'MME': (MMEDataset, MMEDataLoader),
    # 后续添加其他数据集
    # 'VQA': (VQADataset, VQADataLoader),
    # 'COCO': (COCODataset, COCODataLoader),
}

def get_dataset_class(dataset_name: str):
    """根据数据集名称获取数据集类"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知的数据集: {dataset_name}。可用的数据集: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name][0]

def get_data_loader_class(dataset_name: str):
    """根据数据集名称获取数据加载器类"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知的数据集: {dataset_name}。可用的数据集: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name][1]

__all__ = [
    'MMEDataset', 'MMEDataLoader', 
    'BaseDataset', 'BaseDataLoader',
    'get_dataset_class', 'get_data_loader_class',
    'DATASET_REGISTRY'
]