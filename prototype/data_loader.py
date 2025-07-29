import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_length=16000):
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 返回随机音频数据，形状为(max_length,)
        return torch.randn(self.max_length), self.labels[idx]

def create_dataloaders(config):
    """创建虚拟数据集用于测试流程"""
    # 创建虚拟训练集 (100个样本)
    train_dataset = AudioDataset(
        ['dummy'] * 100,
        [0] * 50 + [1] * 50,  # 二分类标签
        max_length=config['max_length']
    )
    
    # 创建虚拟验证集 (20个样本)
    val_dataset = AudioDataset(
        ['dummy'] * 20,
        [0] * 10 + [1] * 10,
        max_length=config['max_length']
    )
    
    # 创建虚拟测试集 (20个样本)
    test_dataset = AudioDataset(
        ['dummy'] * 20,
        [0] * 10 + [1] * 10,
        max_length=config['max_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    return train_loader, val_loader, test_loader