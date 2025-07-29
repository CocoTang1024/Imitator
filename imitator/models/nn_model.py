import os  # 导入操作系统模块，用于环境变量和路径处理
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块及层定义
import math  # 导入数学运算模块
from imitator.models.wav2vec import Wav2Vec2Model  # 导入 Wav2Vec2 预训练模型
from collections import defaultdict  # 导入字典工厂函数，用于默认值

class struct(object):  # 定义一个结构体类，用于动态属性设置
    def __init__(self, **kwargs):  # 构造函数，接收任意关键字参数
        for key, val in kwargs.items():  # 遍历所有参数
            setattr(self, key, val)  # 将参数设置为实例属性

# 定义非线性激活函数，支持 swish 和 relu 两种模式
def nonlinearity(x, activation="swish"):  # x 为输入张量，activation 为激活类型
    if activation == "swish":  # 如果选择 swish
        x = x * torch.sigmoid(x)  # swish 计算：x * sigmoid(x)
    elif activation == "relu":  # 如果选择 relu
        x = torch.relu(x)  # ReLU 激活：max(0, x)
    return x  # 返回激活后的结果

# 获取就地激活层，减少内存拷贝
def get_inplace_activation(activation):  # activation 为激活名称字符串
    if activation == "relu":  # ReLU
        return nn.ReLU(True)  # 就地 ReLU
    elif activation == "leakyrelu":  # LeakyReLU
        return nn.LeakyReLU(0.01, True)  # 负斜率为0.01的 LeakyReLU
    elif activation == "swish":  # swish
        return nn.SiLU(True)  # SiLU 层等价于 swish
    elif activation == "sigmoid":  # sigmoid
        return nn.Sigmoid()  # Sigmoid 层
    else:  # 如果未识别
        raise("Error: Invalid activation")  # 抛出错误

# 根据归一化类型构造对应层
def Normalize(in_channels, norm="batch"):  # in_channels 为通道数，norm 为归一化类型
    if norm == "batch":  # BatchNorm with affine
         return nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=True)  # 一维批归一化
    if norm == "batchfalse":  # BatchNorm without affine
         return nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=False)  # 不可学习参数
    elif norm == "instance":  # InstanceNorm
        return nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)  # 组归一化实例模式
    elif norm == "instancefalse":  # InstanceNorm without affine
        return nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=False)  # 不可学习参数
    else:  # 未识别类型
        raise("Enter a valid norm")  # 抛出错误

# 向层列表中添加归一化层
def add_norm(norm:str, in_channels:int, enc_layers:list):  # norm 类型，in_channels 通道数，enc_layers 层列表
    if norm is not None:  # 如果指定了归一化类型
        fn = Normalize(in_channels, norm)  # 获取归一化层
        enc_layers.append(fn)  # 添加到列表中

# 向层列表中添加激活层
def add_activation(activation:str, layers:list):  # activation 名称，layers 层列表
    if activation is not None:  # 如果指定了激活类型
        layers.append(get_inplace_activation(activation))  # 添加就地激活层

# 默认值工厂函数，返回 None
def def_value():
    return None  # 用于 defaultdict 初始化

# 位置编码模块，给序列位置添加正余弦编码
class PositionalEncoding(nn.Module):  # 继承自 nn.Module
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):  # d_model 编码维度，dropout 丢弃率，max_len 最大长度
        super().__init__()  # 调用父类构造函数
        self.dropout = nn.Dropout(p=dropout)  # 丢弃层

        position = torch.arange(max_len).unsqueeze(1)  # 生成位置索引，形状 [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 计算频率衰减项
        pe = torch.zeros(max_len, 1, d_model)  # 初始化位置编码矩阵
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        self.register_buffer('pe', pe)  # 注册为 buffer，训练中不更新

    def forward(self, x):  # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]  # 加上位置编码
        return self.dropout(x)  # 返回丢弃后的结果

# 构造 encoder-decoder 交互掩码
def enc_dec_mask(device, dataset, T, S):  # device 设备，dataset 数据集名称，T 目标长度，S 源长度
    mask = torch.ones(T, S)  # 初始化全 1 掩码
    if dataset == "BIWI":  # BIWI 数据集自定义对齐
        for i in range(T):  # 遍历时间步
            mask[i, i*2:i*2+2] = 0  # 屏蔽对应位置
    elif dataset == "vocaset":  # vocaset 数据集
        for i in range(T):
            mask[i, i] = 0  # 对角线位置屏蔽
    return (mask == 1).to(device=device)  # 转换为布尔掩码并移到指定设备

# 构造多头自注意力的因果掩码
def casual_mask(n_head, max_seq_len, period):  # n_head 头数，max_seq_len 最大序列长度，period 未使用忽略
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)  # 生成上三角并转置
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # 填充 -inf 和 0
    mask = mask.unsqueeze(0).repeat(n_head, 1, 1)  # 扩展到多头形状 [n_head, L, L]
    return mask  # 返回掩码张量

# 定义运动解码器，用于将特征映射到顶点空间
class motion_decoder(nn.Module):  # 继承自 nn.Module
    def __init__(self,
                 in_channels,  # 输入特征维度
                 out_channels,  # 输出顶点维度
                 activation="leakyrelu",  # 激活类型
                 num_dec_layers=1,  # 解码层数
                 fixed_channel=True,  # 通道数是否固定
                 style_concat=False,  # 是否拼接风格 embedding
                 **ignore_args):  # 忽略其他参数
        super().__init__()  # 父类初始化
        self.style_concat = style_concat  # 保存标志
        if style_concat:  # 如果拼接风格
            in_channels = 2 * in_channels  # 输入通道数翻倍

        if num_dec_layers == 1:  # 单层解码
            self.decoder = nn.Sequential()  # 创建空序列
            final_out_layer = nn.Linear(in_channels, out_channels)  # 最后一层线性
            self.decoder.add_module("final_out_layer", final_out_layer)  # 添加到序列
        else:  # 多层解码
            dec_layers = []  # 层列表
            if fixed_channel:
                ch = in_channels  # 通道数固定
                ch_multi = 1  # 通道扩展系数
            else:
                ch = (out_channels - in_channels) // 2**(num_dec_layers-1)  # 计算初始通道
                ch_multi = 2  # 扩展系数

            dec_layers.append(nn.Linear(in_channels, ch))  # 第一层线性
            add_activation(activation, dec_layers)  # 添加激活层

            for i in range(2, num_dec_layers):  # 其余层
                dec_layers.append(nn.Linear(ch, ch_multi * ch))  # 线性层
                add_activation(activation, dec_layers)  # 添加激活
                ch = ch_multi * ch  # 更新通道数

            decoder = nn.Sequential(*dec_layers)  # 创建子序列
            self.decoder = nn.Sequential(*decoder)  # 包裹为 Sequential

            final_out_layer = nn.Linear(ch, out_channels)  # 最后输出层
            self.decoder.add_module("final_out_layer", final_out_layer)  # 添加

        self.init_weight()  # 初始化权重

    def init_weight(self):  # 权重初始化方法
        for name, param in self.decoder.named_parameters():  # 遍历参数
            if 'bias' in name or "final_out_layer" in name:  # 偏置或最后一层
                nn.init.constant_(param, 0)  # 初始化为 0
            else:
                nn.init.xavier_uniform_(param)  # Xavier 均匀分布初始化

    def forward(self, gen_viseme_feat, style_emb):  # 前向方法
        if self.style_concat:  # 如果拼接风格
            Bs, nf, featdim = gen_viseme_feat.shape  # 批大小、帧数、特征维度
            style_emb = style_emb.repeat(1, nf, 1)  # 重复风格 embedding
            vertice_out_w_style = torch.cat([gen_viseme_feat, style_emb], dim=-1)  # 拼接
        else:
            vertice_out_w_style = gen_viseme_feat + style_emb  # 相加融合

        return self.decoder(vertice_out_w_style)  # 解码并返回

# 主模型：根据音频生成顶点序列
class imitator(nn.Module):  # 继承自 nn.Module
    def __init__(self, **args):  # 接收任意参数
        super(imitator, self).__init__()  # 父类初始化
        if isinstance(args, dict):  # 如果传入 dict
            args = struct(**args)  # 转为 struct 实例

        self.train_subjects = args.train_subjects.split(" ")  # 列表形式的训练主体
        self.dataset = args.dataset  # 数据集名称

        # 加载 Wav2Vec2 音频编码器
        if os.getenv('WAV2VEC_PATH'):  # 如果环境变量中指定路径
            wav2vec_path = os.getenv('WAV2VEC_PATH')  # 读取
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path)  # 从本地加载模型
        elif hasattr(args, 'wav2vec_model'):  # 如果参数中指定模型文件
            wav2vec_path = os.path.join(os.getenv('HOME'), args.wav2vec_model)  # 拼接路径
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path)  # 加载
        else:
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  # 默认加载预训练模型

        if not hasattr(args, 'max_seq_len'):  # 如果未指定最大序列长度
            args.max_seq_len = 600  # 默认 600

        self.audio_encoder.feature_extractor._freeze_parameters()  # 冻结特征提取层
        self.audio_feature_map = nn.Linear(768, args.feature_dim)  # 特征映射到目标维度
        if hasattr(args, 'wav2vec_static_features'):  # 可选静态特征
            self.audio_encoder.generate_static_audio_features = args.wav2vec_static_features  # 设置标志

        self.PPE = PositionalEncoding(args.feature_dim, max_len=args.max_seq_len)  # 位置编码器
        self.causal_mh_mask = casual_mask(n_head=4, max_seq_len=args.max_seq_len, period=args.max_seq_len)  # 因果多头掩码
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim, nhead=4,
            dim_feedforward=2 * args.feature_dim,
            batch_first=True
        )  # 解码器层定义
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)  # Transformer 解码器

        self.obj_vector = nn.Linear(args.num_identity_classes, args.feature_dim, bias=False)  # 身份到风格 embedding
        self.transformer_ff_features = defaultdict(def_value)  # 缓存 Transformer 特征
        self.args = args  # 保存参数

        self.vertice_map_r = motion_decoder(
            in_channels=args.feature_dim,  # 输入特征维度
            out_channels=args.vertice_dim,  # 输出顶点维度
            num_dec_layers=args.num_dec_layers,  # 解码层数
            fixed_channel=args.fixed_channel,  # 通道是否固定
            style_concat=args.style_concat  # 是否拼接风格
        )  # 顶点映射解码器实例

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):  # 前向训练方法
        self.device = audio.device  # 记录设备
        template = template.unsqueeze(1)  # 为模板添加时间维度
        obj_embedding = self.obj_vector(one_hot)  # one-hot 到风格 embedding
        frame_num = vertice.shape[1]  # 序列长度
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state  # 音频编码
        hidden_states = self.audio_feature_map(hidden_states)  # 特征映射

        for i in range(frame_num):  # 逐帧生成
            if i == 0:  # 初始帧
                style_emb = obj_embedding.unsqueeze(1)  # 扩展时间维度
                start_token = torch.zeros_like(style_emb)  # 初始 token
                vertice_input = self.PPE(start_token)  # 位置编码
                vertice_emb = start_token  # 保存输入
            else:
                vertice_input = self.PPE(vertice_emb)  # 追加位置编码

            tgt_mask = self.causal_mh_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)  # 解码器自注意力掩码
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])  # 编解码器掩码
            gen_viseme_feat = self.transformer_decoder(
                vertice_input, hidden_states,
                tgt_mask=tgt_mask, memory_mask=memory_mask
            )  # Transformer 解码
            new_output = gen_viseme_feat[:, -1, :].unsqueeze(1)  # 取最后一步输出
            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)  # 拼接为下一步输入

        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)  # 解码顶点并添加风格
        vertice_out_w_style = vertice_out_w_style + template  # 加回模板变换
        loss = criterion(vertice_out_w_style, vertice)  # 计算损失
        loss = torch.mean(loss)  # 平均损失

        self.batch_viseme_feats = gen_viseme_feat  # 保存特征
        return loss, vertice_out_w_style  # 返回损失与输出

    def style_forward(self, audio, seq_name, template, vertice, one_hot, criterion, teacher_forcing=False):  # 风格前向方法
        assert len(one_hot.shape) == 2  # one-hot 必须为二维
        self.device = audio.device  # 记录设备
        train_from_scratch = teacher_forcing  # 是否重新训练特征
        if self.transformer_ff_features[seq_name] is None or train_from_scratch:  # 未缓存或要求重算
            template = template.unsqueeze(1)  # 添加维度
            frame_num = vertice.shape[1]  # 序列长度
            hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state  # 编码
            hidden_states = self.audio_feature_map(hidden_states)  # 映射

            for i in range(frame_num):  # 逐帧递归
                if i == 0:
                    obj_embedding = self.obj_vector(one_hot)  # one-hot 到 embedding
                    style_emb = obj_embedding.unsqueeze(1)  # 添加维度
                    start_token = torch.zeros_like(style_emb)  # 初始 token
                    vertice_input = self.PPE(start_token)  # 编码
                    vertice_emb = start_token  # 初始 emb
                else:
                    vertice_input = self.PPE(vertice_emb)  # 后续 emb 编码

                tgt_mask = self.causal_mh_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)  # 掩码
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])  # 掩码
                gen_viseme_feat = self.transformer_decoder(
                    vertice_input, hidden_states,
                    tgt_mask=tgt_mask, memory_mask=memory_mask
                )  # 解码
                new_output = gen_viseme_feat[:, -1, :].unsqueeze(1)  # 最后一步
                vertice_emb = torch.cat((vertice_emb, new_output), dim=1)  # 拼接

            self.transformer_ff_features[seq_name] = gen_viseme_feat.detach()  # 缓存特征
        else:
            style_emb = self.obj_vector(one_hot).unsqueeze(1)  # 直接使用 cached embedding

        gen_viseme_feat = self.transformer_ff_features[seq_name]  # 读取缓存
        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)  # 解码
        vertice_out_w_style = vertice_out_w_style + template  # 加模板
        loss = criterion(vertice_out_w_style, vertice)  # 损失
        loss = torch.mean(loss)  # 平均
        return loss, vertice_out_w_style  # 返回

    def predict(self, audio, template, one_hot, test_dataset=None):  # 推理接口
        test_dataset = self.dataset if test_dataset is None else test_dataset  # 选择数据集
        self.device = audio.device  # 设备
        template = template.unsqueeze(1)  # 添加维度
        obj_embedding = self.obj_vector(one_hot)  # one-hot 到 embedding
        hidden_states = self.audio_encoder(audio, test_dataset).last_hidden_state  # 编码
        frame_num = hidden_states.shape[1]  # 序列长
        hidden_states = self.audio_feature_map(hidden_states)  # 映射

        for i in range(frame_num):  # 递归生成
            if i == 0:
                style_emb = obj_embedding.unsqueeze(1)  # 添加维度
                start_token = torch.zeros_like(style_emb)  # 初始 token
                vertice_input = self.PPE(start_token)  # 编码
                vertice_emb = start_token  # emb
            else:
                vertice_input = self.PPE(vertice_emb)  # emb 编码
            tgt_mask = self.causal_mh_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)  # 掩码
            memory_mask = enc_dec_mask(self.device, test_dataset, vertice_input.shape[1], hidden_states.shape[1])  # 掩码
            gen_viseme_feat = self.transformer_decoder(
                vertice_input, hidden_states,
                tgt_mask=tgt_mask, memory_mask=memory_mask
            )  # 解码
            new_output = gen_viseme_feat[:, -1, :].unsqueeze(1)  # 输出
            vertice_emb = torch.cat((vertice_emb, new_output), dim=1)  # 拼接

        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)  # 解码
        vertice_out = vertice_out_w_style + template  # 加模板
        return vertice_out  # 返回最终顶点输出
