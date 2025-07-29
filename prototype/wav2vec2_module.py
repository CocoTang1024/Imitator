import torch
import pytorch_lightning as pl
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2Lightning(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化wav2vec2模型
        config = Wav2Vec2Config()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, audio_input):
        # 提取音频特征
        outputs = self.wav2vec2(audio_input)
        hidden_states = outputs.last_hidden_state
        
        # 全局平均池化
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)
    
    def training_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        audio, labels = batch
        logits = self(audio)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log('test_acc', acc, prog_bar=True)
        return acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }