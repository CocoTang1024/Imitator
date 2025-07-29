import yaml
import os
import torch

default_config = {
    'model': {
        'num_classes': 10,
        'learning_rate': 1e-4
    },
    'data': {
        'max_length': 16000,
        'batch_size': 32
    },
    'trainer': {
        'max_epochs': 10,
        'gpus': 1 if torch.cuda.is_available() else 0
    }
}

def load_config(config_path):
    """从YAML文件加载配置"""
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, using default config")
        return default_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 合并默认配置和文件配置
    merged_config = {**default_config, **config}
    return merged_config