import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from config import load_config
from data_loader import create_dataloaders
from wav2vec2_module import Wav2Vec2Lightning

if __name__ == '__main__':
    # 加载配置
    config = load_config('config.yaml')  # 默认使用当前目录下的config.yaml
    
    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(config['data'])
    
    # 初始化模型
    model = Wav2Vec2Lightning(
        num_classes=config['model']['num_classes'],
        learning_rate=config['model']['learning_rate']
    )
    
    # 设置日志记录（保存在prototype目录下）
    logger = TensorBoardLogger('lightning_logs', name='wav2vec2')
    
    # 配置训练器
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        gpus=config['trainer']['gpus'],
        logger=logger,
        log_every_n_steps=10,
        progress_bar_refresh_rate=1
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    print('Training completed!')