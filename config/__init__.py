import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class DataConfig:
    dataset_name: str
    data_path: str
    train_split: float
    val_split: float
    test_split: float
    batch_size: int
    num_workers: int
    max_length: int
    image_size: int

@dataclass
class TokenPruningConfig:
    enabled: bool
    method: str
    prune_ratio: float

@dataclass
class TokenFusionConfig:
    enabled: bool
    method: str
    fusion_threshold: float

@dataclass
class ModelConfig:
    base_model: str
    token_pruning: TokenPruningConfig
    token_fusion: TokenFusionConfig

@dataclass
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int
    max_grad_norm: float
    eval_steps: int
    save_steps: int

@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

class ConfigManager:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name=self.config['experiment']['name'],
            output_dir=self.config['experiment']['output_dir'],
            seed=self.config['experiment']['seed'],
            device=self.config['experiment']['device'],
            data=DataConfig(**self.config['data']),
            model=ModelConfig(
                base_model=self.config['model']['base_model'],
                token_pruning=TokenPruningConfig(**self.config['model']['token_pruning']),
                token_fusion=TokenFusionConfig(**self.config['model']['token_fusion'])
            ),
            training=TrainingConfig(**self.config['training'])
        )