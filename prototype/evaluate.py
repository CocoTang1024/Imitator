import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import load_config
from data_loader import create_dataloaders
from wav2vec2_module import Wav2Vec2Lightning

def evaluate_model(model, dataloader, device):
    """在指定数据加载器上评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device)
            labels = labels.to(device)
            
            logits = model(audio)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds

if __name__ == '__main__':
    # 加载配置
    config = load_config('config.yaml')
    
    # 创建数据加载器
    _, val_loader, _ = create_dataloaders(config['data'])
    
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
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 运行评估
    true_labels, pred_labels = evaluate_model(model, val_loader, device)
    
    # 输出评估结果
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    
    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    print(f"\nValidation Accuracy: {accuracy:.4f}")