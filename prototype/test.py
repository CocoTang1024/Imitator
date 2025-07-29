import torch
import pytorch_lightning as pl
from config import load_config
from data_loader import create_dataloaders
from wav2vec2_module import Wav2Vec2Lightning

if __name__ == '__main__':
    # 加载配置
    config = load_config('config.yaml')
    
    # 创建数据加载器
    _, _, test_loader = create_dataloaders(config['data'])
    
    import os
    import glob
    
    # 查找prototype目录下的检查点文件
    checkpoint_files = glob.glob('prototype/logs/wav2vec2/version_*/checkpoints/*.ckpt')
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint found. Please train the model first.")
    
    # 获取最新修改的检查点文件
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    
    # 加载训练好的模型
    model = Wav2Vec2Lightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # 配置测试器
    trainer = pl.Trainer(
        gpus=config['trainer']['gpus'],
        logger=False,
        progress_bar_refresh_rate=1
    )
    
    # 运行测试
    test_result = trainer.test(model, test_loader)
    print(f'Test accuracy: {test_result[0]["test_acc"]:.4f}')