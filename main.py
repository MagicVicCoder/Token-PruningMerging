import os
import torch
import argparse
import sys

from config import ConfigManager
from train import Trainer
from test import test_model
from utils.downloader import DatasetManager

# 设置 Hugging Face 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """检查环境依赖"""
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training will be slow on CPU.")
    
    # 检查必要的Python包
    try:
        import transformers
        import PIL
        import numpy
        print("All required packages are available.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install all requirements: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description="Qwen-VL Token Pruning and Fusion")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], 
                       default="both", help="运行模式")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                       help="配置文件路径")
    parser.add_argument("--create_splits", action="store_true", 
                       help="创建数据集划分")
    parser.add_argument("--download_data", action="store_true",
                       help="下载数据集（如果不存在）")
    parser.add_argument("--skip_download", action="store_true",
                       help="跳过自动下载检查")
    
    args = parser.parse_args()
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    config = config_manager.get_experiment_config()
    
    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 下载数据（如果需要）
    if not args.skip_download:
        print("Checking dataset and model availability...")
        
        # 检查并准备数据集
        if not DatasetManager.ensure_dataset_ready(config):
            if args.download_data:
                print("Dataset download failed. Please check the error messages above.")
                sys.exit(1)
            else:
                print("Dataset is not ready. Use --download_data to attempt automatic download.")
                print(f"Or manually prepare the dataset at {config.data.data_path}")
                sys.exit(1)
        
        # 检查并准备模型
        if not DatasetManager.ensure_model_ready(config):
            print("Model is not ready. Please check the error messages above.")
            sys.exit(1)
    
    # 创建数据集划分（如果需要）
    if args.create_splits:
        from data.dataset import create_dataset_splits
        create_dataset_splits(config.data.data_path, 
                             config.data.train_split, 
                             config.data.val_split)
        print("数据集划分创建完成")
        return
    
    # 根据模式运行
    if args.mode in ["train", "both"]:
        print("开始训练...")
        trainer = Trainer(config)
        trainer.train()
    
    if args.mode in ["test", "both"]:
        print("开始测试...")
        test_results = test_model()
        print("\n最终测试结果:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
