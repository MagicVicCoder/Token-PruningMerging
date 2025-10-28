'''
import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import hashlib
from typing import Optional
from huggingface_hub import snapshot_download
import torch

class DataDownloader:
    """数据下载器"""
    
    @staticmethod
    def download_model(model_name: str, cache_dir: Optional[str] = None):
        """下载模型"""
        print(f"Downloading model {model_name}...")
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True
            )
            print(f"Model {model_name} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            raise

class DatasetManager:
    """数据集管理器"""
    
    @staticmethod
    def ensure_dataset_ready(config):
        """确保数据集已准备就绪 - 现在使用Hugging Face datasets"""
        print("MME dataset will be loaded from Hugging Face Hub using datasets library")
        
        # 测试是否可以加载数据集
        try:
            from datasets import load_dataset
            print("Testing MME dataset loading...")
            # 尝试加载一个小样本进行测试
            dataset = load_dataset("lmms-lab/MME")
            print(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
            return True
        except Exception as e:
            print(f"Error loading MME dataset: {e}")
            print("Please check:")
            print("1. Internet connection")
            print("2. Dataset availability at: https://huggingface.co/datasets/lmms-lab/MME")
            print("3. You might need to accept the terms of use on the dataset page")
            return False
    
    @staticmethod
    def ensure_model_ready(config):
        """确保模型已准备就绪"""
        try:
            # 尝试加载模型，如果失败会自动下载
            from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
            
            print(f"Loading processor for {config.model.base_model}...")
            processor = Qwen2VLProcessor.from_pretrained(config.model.base_model)
            
            print(f"Loading model {config.model.base_model}...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.model.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print("Model loaded successfully.")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("This might be due to missing model files.")
            
            try:
                # 尝试下载模型
                DataDownloader.download_model(config.model.base_model)
                return True
            except Exception as download_error:
                print(f"Failed to download model: {download_error}")
                return False
'''
import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import hashlib
from typing import Optional
from huggingface_hub import snapshot_download
import torch

class DataDownloader:
    """数据下载器"""
    
    @staticmethod
    def download_model(model_name: str, cache_dir: Optional[str] = None):
        """下载模型"""
        print(f"Downloading model {model_name}...")
        try:
            # 使用镜像站下载模型
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True
            )
            print(f"Model {model_name} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            raise

class DatasetManager:
    """数据集管理器"""
    
    @staticmethod
    def ensure_dataset_ready(config):
        """确保数据集已准备就绪 - 现在使用Hugging Face datasets"""
        print("MME dataset will be loaded from Hugging Face Hub using datasets library")
        
        # 确保使用镜像站
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print("已设置 Hugging Face 镜像站: hf-mirror.com")
        
        # 增加超时时间
        os.environ['HF_DATASETS_TIMEOUT'] = '300'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
        
        # 测试是否可以加载数据集
        try:
            from datasets import load_dataset
            print("Testing MME dataset loading...")
            # 使用镜像站加载数据集
            dataset = load_dataset("lmms-lab/MME")
            print(f"Dataset loaded successfully. Available splits: {list(dataset.keys())}")
            return True
        except Exception as e:
            print(f"Error loading MME dataset: {e}")
            print("Please check:")
            print("1. Internet connection")
            print("2. Dataset availability at: https://hf-mirror.com/datasets/lmms-lab/MME")
            print("3. You might need to accept the terms of use on the dataset page")
            return False
