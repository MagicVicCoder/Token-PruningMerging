from .mme_loader import MMEDataset, MMEDataLoader
from .base_loader import BaseDataset, BaseDataLoader

def explore_dataset():
    """探索数据集结构"""
    from datasets import load_dataset
    try:
        dataset = load_dataset("lmms-lab/MME", trust_remote_code=True)
        print("Dataset structure:")
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"Sample keys: {list(sample.keys())}")
                break
    except Exception as e:
        print(f"Error exploring dataset: {e}")

__all__ = ['MMEDataset', 'MMEDataLoader', 'BaseDataset', 'BaseDataLoader', 'explore_dataset']