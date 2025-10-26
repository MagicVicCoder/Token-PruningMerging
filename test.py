import os
import torch
from config import ConfigManager
from data import get_data_loader_class
from model.pruned_model import QwenVLPrunedModel
from evaluator.evaluator import ModelEvaluator
from utils.logger import TrainingLogger
from utils.downloader import DatasetManager

def test_model():
    """测试模型"""
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get_experiment_config()
    
    # 初始化日志记录器
    logger = TrainingLogger(config.output_dir)
    
    # 确保数据和模型已准备
    if not DatasetManager.ensure_dataset_ready(config):
        raise RuntimeError("Dataset is not ready. Please check the dataset path.")
    
    # 加载数据
    logger.log_info("Loading test data...")
    data_loader_class = get_data_loader_class(config.data.dataset_name)
    data_loader = data_loader_class(config)
    _, _, test_loader = data_loader.get_data_loaders()
    
    # 加载模型
    logger.log_info("Loading model...")
    model = QwenVLPrunedModel(config, config.model)
    
    # 加载训练好的权重（如果存在）
    model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.log_info(f"Loaded trained model from: {model_path}")
    else:
        logger.log_info("Using untrained model for testing")
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 评估模型
    evaluator = ModelEvaluator(model, device, logger)
    test_metrics = evaluator.evaluate(test_loader, "Test")
    
    # 保存详细结果
    results_path = os.path.join(config.output_dir, "detailed_test_results.txt")
    with open(results_path, 'w') as f:
        f.write("Detailed Test Results:\n")
        f.write("=" * 50 + "\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    logger.log_info(f"Detailed test results saved to: {results_path}")
    logger.close()
    
    return test_metrics

if __name__ == "__main__":
    test_results = test_model()
    print("Test Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")