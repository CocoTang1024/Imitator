import torch                                   # PyTorch 主库，张量及自动求导
import torch.nn as nn                          # nn 子模块，提供各类层与损失
import torch.nn.functional as F                # 功能性 API，包含常用激活函数、插值等
import numpy as np                             # NumPy，用于 CPU 端向量/矩阵运算
import copy                                    # Python 内建模块，用于对象深拷贝
import math                                    # Python 数学库，提供常量与函数

# Transformer/Wav2Vec2.0 相关依赖
from transformers import Wav2Vec2Model, Wav2Vec2Config          # 预训练模型与配置
from transformers.modeling_outputs import BaseModelOutput       # HuggingFace 统一输出格式
from typing import Optional, Tuple                              # 类型标注

_CONFIG_FOR_DOC = "Wav2Vec2Config"  # 文档用常量，指示该实现对应的配置类

# ------------------------------------------------------------------------------
# 说明：以下 Wav2Vec2Model 的大量实现摘自 HuggingFace 官方源码
#       https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
#       在此基础上做少量修改以满足音频驱动人脸动画的需求
# ------------------------------------------------------------------------------

# -------------------------- 辅助函数：随机时间掩码 -----------------------------
def _compute_mask_indices(
    shape: Tuple[int, int],                   # (batch_size, 序列长度)
    mask_prob: float,                         # 掩码比例
    mask_length: int,                         # 单次掩码连续帧长度
    attention_mask: Optional[torch.Tensor] = None,  # 真实音频长度掩码，排除 padding
    min_masks: int = 0,                       # 最少掩码块数
) -> np.ndarray:
    """
    生成 SpecAugment 的时间维掩码索引。

    返回：
        mask (ndarray[bool]) → shape 与输入相同，True 代表需要被掩码
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)      # 先全部设为 False

    # 计算理论掩码块数（向上取整 + 随机）
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []                            # 每个 batch 的掩码索引列表
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None

    for i in range(bsz):
        if padding_mask is not None:          # 若有有效长度掩码
            sz = all_sz - padding_mask[i].long().sum().item()      # 去掉 padding 的真实长度
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)                   # 每段长度默认 mask_length

        # 边界情况：若总掩码长度为 0，则至少掩 1 帧
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        # 如果剩余长度不足，则缩短最小长度
        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        # 随机选择起始位置
        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        # 把连续长度展开
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))       # 去重并限长

    # 保证每个 batch 掩码长度一致（取最小）
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask

# ------------------------- 辅助函数：线性插值重采样 ------------------------------
def linear_interpolation(features: torch.Tensor,
                         input_fps: int,
                         output_fps: int,
                         output_len: Optional[int] = None):
    """
    对特征序列进行线性插值，改变采样率。

    参数：
        features   : (B, T, C) 或 (B, C, T) 这里假定第二种 (B, C, T)
        input_fps  : 原始帧率
        output_fps : 目标帧率
        output_len : 若指定则直接插值到该长度，否则按帧率比例计算
    返回：
        插值后的特征 (B, T', C) 与输入模式一致 (这里返回 (B, T', C))
    """
    features = features.transpose(1, 2)                       # 变为 (B, C, T)
    seq_len = features.shape[2] / float(input_fps)            # 原序列时长 (秒)
    if output_len is None:
        output_len = int(seq_len * output_fps)                # 目标帧数 = 时长 * 目标 fps
    output_features = F.interpolate(features,                 # 线性插值
                                     size=output_len,
                                     align_corners=True,
                                     mode='linear')
    return output_features.transpose(1, 2)                    # 再转回 (B, T', C)

# ----------------------------- 主模型重写 --------------------------------------
class Wav2Vec2Model(Wav2Vec2Model):  # 继承并“猴子补丁”式地覆写官方实现
    def __init__(self, config: Wav2Vec2Config):
        """
        仅做少量扩展：如静态特征开关 generate_static_audio_features。
        其余部分复用父类初始化逻辑（即加载预训练权重）。
        """
        super().__init__(config)
        self.generate_static_audio_features = False          # 若为 True，则 forward 时转 eval() 模式

    # ------------- 前向传播（主要改动集中在数据预处理与插值） -------------------
    def forward(
        self,
        input_values: torch.Tensor,      # 原始语音波形 (B, T) [-1,1]
        dataset: str,                    # 数据集名称（控制不同采样策略）
        attention_mask: Optional[torch.Tensor] = None,       # 有效长度掩码
        output_attentions: Optional[bool] = None,            # 是否返回注意力
        output_hidden_states: Optional[bool] = None,         # 是否返回隐藏层
        return_dict: Optional[bool] = None,                  # 是否用字典输出
        frame_num: Optional[int] = None,                     # 目标序列帧数（插值用）
        fps: Optional[int] = None                            # 暂留字段，未使用
    ):
        # 测试阶段若想获得稳定特征，可开关 generate_static_audio_features
        if self.generate_static_audio_features:
            self.eval()                                      # 进入 eval，关闭 dropout

        # 若未显式指定，则沿用 config 中的默认设置
        self.config.output_attentions = True                 # 强制让 encoder 返回注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) 特征提取器：Conv1D 模块，降采样并提取局部特征
        hidden_states = self.feature_extractor(input_values) # (B, C, T')
        hidden_states = hidden_states.transpose(1, 2)        # 变为 (B, T', C)

        # ----------------------- 数据集相关的对齐逻辑 ---------------------------
        if dataset == "BIWI":
            # BIWI 下：特征下采样 20ms 一帧。若长度为奇数，裁掉最后 1 帧以保证偶数帧；
            # 若给定 frame_num，且特征过长，则截断
            if hidden_states.shape[1] % 2 != 0:
                hidden_states = hidden_states[:, :-1]
            if frame_num and hidden_states.shape[1] > frame_num * 2:
                hidden_states = hidden_states[:, :frame_num * 2]

        elif dataset == "vocaset":
            # Vocaset 语音 50fps → 30fps，对应人脸序列 30fps，使用线性插值
            hidden_states = linear_interpolation(hidden_states,
                                                 input_fps=50,
                                                 output_fps=30,
                                                 output_len=frame_num)

        elif dataset == "ardzdf":
            # ARDZDF 同 vocaset，50 → 30fps
            hidden_states = linear_interpolation(hidden_states,
                                                 input_fps=50,
                                                 output_fps=30,
                                                 output_len=frame_num)

        # ----------------------------------------------------------------------
        # attention_mask 需同步下采样：将样本级 mask 转为 Conv 输出级别
        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            # 重新生成下采样后的掩码，1 表示有效帧
            attention_mask = torch.zeros(hidden_states.shape[:2],
                                         dtype=hidden_states.dtype,
                                         device=hidden_states.device)
            # 标记每条序列最后一个有效位置
            attention_mask[(torch.arange(attention_mask.shape[0], device=hidden_states.device),
                            output_lengths - 1)] = 1
            # 反向累积得到完整 mask
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 2) 特征投影：LayerNorm + Linear，把 conv 特征映射到 Transformer 维度
        hidden_states = self.feature_projection(hidden_states)[0]  # 投影后 shape (B, T', D)

        # -------------------------- SpecAugment (训练态) ------------------------
        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.size()

            # 时间维掩码
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)

            # 特征维掩码
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0

        # 3) Transformer Encoder
        encoder_outputs = self.encoder(
            hidden_states,                          # 输入 (B, T', D)
            attention_mask=attention_mask,          # 变长掩码
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]         # 主输出 last_hidden_state

        # 若选择 tuple 输出格式
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 否则封装为 BaseModelOutput，方便下游统一接口
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
