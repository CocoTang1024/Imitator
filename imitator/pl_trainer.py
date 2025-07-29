import pytorch_lightning as pl  # 引入 PyTorch Lightning，高层封装训练/验证/测试循环，简化分布式训练等
import numpy as np              # 引入 NumPy，主要用于 CPU 端的张量/数组计算和统计
import torch                    # 引入 PyTorch，深度学习核心库
import torch.nn as nn           # 引入 nn 子模块，包含神经网络常用层与损失函数
from imitator.utils.init_from_config import instantiate_from_config  # 根据配置动态实例化模型或组件的工具函数
from FLAMEModel.flame_masks import get_flame_mask                    # 读取 FLAME 头模各部位顶点索引的函数
from imitator.utils.losses import Custom_errors                      # 自定义复合损失类，封装多种误差度量
from imitator.models.nn_model import imitator                        # 具体的 IMITATOR 网络结构

class Imitator(pl.LightningModule):                                  # 继承 LightningModule，用于组织训练逻辑
    def __init__(self,
                 optim_params,                                       # 优化器与 scheduler 超参数字典
                 monitor,                                            # 监控指标名称（如 val_loss）
                 nn_model_cfg,                                       # 用于实例化骨干网络的配置
                 loss_cfg                                            # 自定义损失各项权重配置
                 ):
        super(Imitator, self).__init__()                             # 初始化父类

        # imitator_no_motion_enc_style_input_to_dec
        self.optim_params = optim_params                             # 保存优化器超参数
        self.nn_model: imitator = instantiate_from_config(nn_model_cfg)  # 动态实例化 imitator 网络

        self.loss = nn.MSELoss()                                     # 基础重建损失：均方误差
        self.monitor = monitor                                       # 记录需要监控的指标名

        # setup teacher forcing train_teacher_forcing
        self.teacher_forcing = nn_model_cfg["params"].get("train_teacher_forcing", False)  # 是否在训练阶段使用教师强制

        # create the custom losses
        self.vertice_dim = self.nn_model.args.vertice_dim            # 单帧顶点维度（V*3）
        # masks for the lips
        mask = get_flame_mask()                                      # 获取不同面部区域的顶点索引
        self.lips_idxs = mask.lips                                   # 取嘴唇区域索引
        lip_mask = torch.zeros((1, self.nn_model.args.vertice_dim // 3, 3))  # 初始化嘴唇掩码 (1, V, 3)
        lip_mask[0, self.lips_idxs] = 1.0                            # 仅嘴唇顶点为 1，其余为 0
        self.lip_mask = lip_mask.view(1, -1)                         # 展平成 (1, V*3) 形状，方便与顶点逐元素相乘
        self.custom_loss = Custom_errors(self.vertice_dim,           # 初始化复合损失类
                                         loss_creterion=self.loss,
                                         loss_dict=loss_cfg)

    def init_from_ckpt(self, path, ignore_keys=list()):              # 从 checkpoint 加载，忽略部分键
        sd = torch.load(path, map_location="cpu")["state_dict"]      # 读取权重字典到 CPU
        keys = list(sd.keys())                                       # 所有键列表
        ignore_keys.extend(["nn_model.PPE.pe"])                      # 额外忽略位置编码权重
        print(keys)                                                  # 打印所有键（调试）

        ignore_keys = set(ignore_keys)                               # 转为集合便于检查
        for k in keys:
            for ik in ignore_keys:
                if ik in k:                                          # 若需要忽略
                    print(f"Deleting key {k} from state_dict.")      # 打印被忽略的键
                    del sd[k]                                        # 删除该键
        # reset the ignore keys
        del ignore_keys                                              # 释放内存
        unmatched_keys = self.load_state_dict(sd, strict=False)      # 允许部分键不匹配
        print(f"\nRestored from {path}\n")                           # 输出恢复路径
        print("unmatched keys", unmatched_keys)                      # 输出未匹配键，帮助排查

    def configure_optimizers(self):                                  # Lightning 特定函数：返回优化器与 LR 调度器
        # sum all the params
        params = list(self.nn_model.parameters())                    # 收集网络可训练参数

         # optim params
        weight_decay = self.optim_params.get("weight_decay", 0.0)    # L2 正则系数
        learning_rate = self.optim_params.get("lr")                  # 基础学习率

        print("running with weight decay", self.optim_params)        # 控制台输出超参数
        print()

        optimizer = torch.optim.Adam(params,                         # 构造 Adam 优化器
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        lr_sch_factor = self.optim_params.get("lr_sch_factor", 0.85) # ReduceLROnPlateau 的衰减因子
        lr_sch_patience = self.optim_params.get("lr_sch_patience", 500)  # 容忍步数
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min',
                            factor=lr_sch_factor, patience=lr_sch_patience,
                            min_lr=1e-9),                            # 最小学习率设定
            'monitor': "train/rec_loss"}                             # 依据训练重建损失调整 LR

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} # Lightning 识别的返回格式

    def get_input(self, batch, batch_idx):                           # 统一解包 batch，便于后续调用
        if self.lip_mask.device != self.device:                      # 确保 lip_mask 与模型同设备
            self.lip_mask = self.lip_mask.to(self.device)

        audio, vertice, template, one_hot_all, file_name = batch     # 解构 batch
        return audio, vertice, template, one_hot_all, file_name

    def training_step(self, batch, batch_idx):                       # Lightning 训练循环中每个 batch 调用
        audio, vertice, template, one_hot, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]                         # 当前序列名（无扩展名）

        # 调用骨干网络前向：返回重建损失与预测顶点序列
        rec_loss, pred_verts = self.nn_model(audio, template, vertice, one_hot,
                                             self.loss, teacher_forcing=self.teacher_forcing)
        mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)  # MBP (masked best pose) 损失
        net_loss = rec_loss + mbp_loss                               # 综合损失

        # 将多种指标写入 TensorBoard/Loss Logger
        self.log("train/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/mbp_loss", mbp_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("train/rec_loss", rec_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        ### velocity loss
        velocity_loss, weighted_velocity_loss = self.custom_loss.velocity_loss(pred_verts, vertice)  # 计算速度一致性损失
        net_loss = net_loss + weighted_velocity_loss              # 加权加入总损失
        self.log("train/velocity_loss", velocity_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        out_dict = {                                               # 收集数值便于离线统计
            "net_loss": net_loss.item(),
            "rec_loss": rec_loss.item(),
            "mbp_loss": mbp_loss.item(),
        }

        predicted_keypoints = pred_verts.clone().detach().cpu().numpy()  # 保存预测关键点到 CPU/NumPy
        gt_kp = vertice.clone().detach().cpu().numpy()                   # GT 关键点

        return {'loss': net_loss,               # Lightning 识别的关键字，驱动 backward()
                'results': out_dict,
                'predicted_kp': predicted_keypoints,
                'gt_kp': gt_kp
                }

    def val_shared_step(self, batch, batch_idx):                     # 训练集/验证集公用逻辑，减少重复
        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]                          # 当前片段名
        train_subject = "_".join(file_name[0].split("_")[:-1])        # 从文件名推断说话人

        if train_subject in self.nn_model.train_subjects:             # 若当前说话人包含在训练列表中
            condition_subject = train_subject
            iter = self.nn_model.train_subjects.index(condition_subject)  # 找到对应 one-hot 索引
            one_hot = one_hot_all[:, iter, :] if len(one_hot_all.size()) == 3 else one_hot_all
            loss, pred_verts = self.nn_model(audio, template, vertice, one_hot, self.loss)  # 前向计算
            mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
            velocity_loss, weighted_velocity_loss = self.custom_loss.velocity_loss(pred_verts, vertice)

        else:                                                          # 未出现的说话人：遍历所有条件求平均
            condition_loss = []
            condition_mbp_loss = []
            condition_velocity_loss = []

            for iter in range(one_hot_all.shape[-1]):                  # 循环不同条件
                one_hot = one_hot_all[:, iter, :]
                loss, pred_verts = self.nn_model(audio, template, vertice, one_hot, self.loss)
                mbp_loss = self.custom_loss.compute_mbp_reconstruction_loss(pred_verts, vertice, subjsen)
                condition_loss.append(loss.item())
                condition_mbp_loss.append(mbp_loss.item())
                velocity_loss, _ = self.custom_loss.velocity_loss(pred_verts, vertice)
                condition_velocity_loss.append(velocity_loss.item())

            loss = np.mean(condition_loss)                             # 多条件平均
            mbp_loss = np.mean(condition_mbp_loss)
            velocity_loss = np.mean(condition_velocity_loss)

        # 统一返回字典，供验证/测试步骤调用
        return {
            "rec_loss": loss,
            "mbp_loss": mbp_loss,
            "pred_verts": pred_verts,
            "velocity_loss": velocity_loss,
        }

    def validation_step(self, batch, batch_idx):                      # Lightning 验证循环
        result_dict = self.val_shared_step(batch, batch_idx)          # 共用前向函数
        net_loss = result_dict["rec_loss"] + result_dict["mbp_loss"]  # 基础损失
        net_loss += self.custom_loss.loss_dict.get("velocity_weight", 0.0) * result_dict["velocity_loss"]  # 加上速度损失

        # 记录各项指标
        self.log("val/rec_loss", result_dict["rec_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/net_loss", net_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/mbp_loss", result_dict["mbp_loss"] /
                 self.custom_loss.mbp_reconstruction_loss.closed_frame_weight,
                 prog_bar=False, logger=True, on_step=False, on_epoch=True)

        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)  # 重新拿 GT
        return {'loss': net_loss,
                'predicted_kp': result_dict["pred_verts"].cpu().numpy(),
                'gt_kp': vertice.cpu().numpy(),
                }

    def test_step(self, batch, batch_idx):                            # 测试阶段——需返回更多指标与预测序列
        audio, vertice, template, one_hot_all, file_name = self.get_input(batch, batch_idx)
        subjsen = file_name[0].split(".")[0]
        train_subject = "_".join(file_name[0].split("_")[:-1])

        result_npy_dict = {}                                          # 保存不同条件下的预测
        loss = []                                                     # 记录重建损失
        full_reconstruction_mm = []                                   # 全脸重建误差（mm）
        lip_reconstruction_mm = []                                    # 嘴唇重建误差
        lip_sync_metric = []                                          # 同步指标（音频对口型）

        if train_subject in self.nn_model.train_subjects:             # 已见说话人
            condition_subject = train_subject
            iter = self.nn_model.train_subjects.index(condition_subject)
            one_hot = one_hot_all[:, iter, :] if len(one_hot_all.size()) == 3 else one_hot_all
            prediction = self.nn_model.predict(audio, template, one_hot)  # 使用预测模式（无教师强制）
            pred_len = prediction.shape[1]                             # 预测帧数
            vertice = vertice[:, :pred_len]                            # 对齐 GT 长度
            loss.append(self.loss(prediction, vertice).item())

            # reconstruction in mm
            full_reconstruction_mm.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
            lip_reconstruction_mm.append(
                self.custom_loss.compute_masked_error_in_mm(prediction, vertice, self.lip_mask).cpu().numpy())
            lip_sync_metric.append(self.custom_loss.lip_sync_metric(prediction, vertice, self.lip_mask).item())

            prediction = prediction.squeeze()  # (seq_len, V*3)       # 去掉 batch 维度
            out_file = file_name[0].split(".")[0] + "_condition_" + condition_subject
            result_npy_dict[out_file] = prediction.detach().cpu().numpy()
        else:                                                          # 未见说话人：对每个条件都预测
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = self.nn_model.train_subjects[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = self.nn_model.predict(audio, template, one_hot)

                pred_len = prediction.shape[1]
                vertice = vertice[:, :pred_len]

                loss.append(self.loss(prediction, vertice).item())
                full_reconstruction_mm.append(self.custom_loss.error_in_mm(prediction, vertice).cpu().numpy())
                lip_reconstruction_mm.append(self.custom_loss.compute_masked_error_in_mm(
                                                prediction, vertice, self.lip_mask).cpu().numpy())
                lip_sync_metric.append(self.custom_loss.lip_sync_metric(prediction, vertice, self.lip_mask).item())

                prediction = prediction.squeeze()  # (seq_len, V*3)
                out_file = file_name[0].split(".")[0] + "_condition_" + condition_subject
                result_npy_dict[out_file] = prediction.detach().cpu().numpy()

        out_dict = {                                                 # 计算所有条件平均指标
            "metric_rec_loss": np.mean(loss),
            "full_reconstruction_mm": np.mean(full_reconstruction_mm),
            "lip_reconstruction_mm": np.mean(lip_reconstruction_mm),
            "lip_sync_metric": np.mean(lip_sync_metric),
        }

        gt_kp = vertice.cpu().numpy().squeeze()                      # GT 顶点序列
        return {'results': out_dict,
                'prediction_dict': result_npy_dict,
                "gt_kp": gt_kp,
                'seq_name': file_name,
                'seq_len': prediction.shape[0]}                      # 返回序列长度便于可视化
