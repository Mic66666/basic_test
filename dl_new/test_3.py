import argparse
import os
import random
import itertools
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    logging.warning(f"无法导入TensorBoard: {str(e)}，可视化功能不可用")
    SummaryWriter = None
except Exception as e:
    logging.error(f"TensorBoard初始化异常: {str(e)}")
    SummaryWriter = None
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import psutil
import GPUtil
from sklearn.utils.class_weight import compute_class_weight

# 添加上级目录到模块搜索路径中（使用相对路径代替硬编码路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from sleep_stage_config import Config
from dataset_builder_loader.data_loader import DataLoader as MyDataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 添加日志功能的设置函数
def setup_logger(log_path):
    """创建独立日志器代替修改根日志器"""
    logger = logging.getLogger("SleepStage")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加处理器
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件处理器
        log_file = os.path.join(log_path, 'experiment.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger



def convert_args_to_str(hparams):
    """
    将超参数字典转换为字符串（用于保存模型文件名中）
    """
    return "_".join([f"{key}={value}" for key, value in sorted(hparams.items())])

def write_arguments_to_file(args, file_path):
    """
    将参数写入文件（方便记录当前实验参数）
    """
    with open(file_path, "w") as f:
        f.write(str(args))

class BiLSTMAttentionNet(nn.Module):
    """
    定义双向LSTM加注意力机制的神经网络模型
    """
    def __init__(self, input_size, hidden_size, lstm_layers, num_classes, dropout=0.5, rnn_type="LSTM", num_heads=4):
        super(BiLSTMAttentionNet, self).__init__()
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=lstm_layers,
                               batch_first=True,
                               bidirectional=True,
                               dropout=dropout if lstm_layers > 1 else 0)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=lstm_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=dropout if lstm_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        # 修改：将注意力层输出维度由 1 改为 hidden_size，并增加 attention_vector 参数
        self.attention_layer = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_vector = nn.Parameter(torch.randn(hidden_size, 1))
        self.dropout = nn.Dropout(dropout)
        # 新增全连接层以及 Batch Normalization 层
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # 新增：多头注意力机制，用于捕获更多维度的注意力信息
        # 确保 embed_dim 可被 num_heads 整除，否则添加投影层
        if (hidden_size * 2) % num_heads == 0:
            self.mha_projector = None
            self.multi_head_attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout)
        else:
            new_embed_dim = (hidden_size * 2 // num_heads) * num_heads  # 保证 new_embed_dim 是 num_heads 的倍数
            self.mha_projector = nn.Linear(hidden_size * 2, new_embed_dim)
            self.mha_output_projection = nn.Linear(new_embed_dim, hidden_size * 2)
            self.multi_head_attn = nn.MultiheadAttention(embed_dim=new_embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 修改辅助分支为通用结构，根据num_classes动态调整
        self.aux_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        ) if num_classes > 2 else None  # 二分类时不使用辅助分支

    def forward(self, x):
        """
        x 的形状: (batch_size, seq_len, input_size)
        """
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size*2)
        
        # 原始注意力计算
        attn_intermediate = torch.tanh(self.attention_layer(rnn_out))
        attn_scores = torch.matmul(attn_intermediate, self.attention_vector)
        attn_scores = attn_scores.squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 多头注意力计算
        if self.mha_projector is not None:
            mha_input = self.mha_projector(rnn_out)  # 投影到 new_embed_dim
        else:
            mha_input = rnn_out
        mha_input_transposed = mha_input.permute(1, 0, 2)  # (seq_len, batch_size, features)
        attn_output, _ = self.multi_head_attn(mha_input_transposed, mha_input_transposed, mha_input_transposed)
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, features)
        if self.mha_projector is not None:
            # 将输出反向投影回 hidden_size*2
            attn_output = self.mha_output_projection(attn_output)
        # 融合注意力机制
        context_original = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)
        context_multi = torch.sum(attn_output * attn_weights.unsqueeze(-1), dim=1)
        context_combined = 0.7 * context_original + 0.3 * context_multi
        
        # 后续处理保持不变
        context_combined = self.dropout(context_combined)
        context_fc = self.fc(context_combined)
        context_bn = self.bn(context_fc)
        context_relu = torch.relu(context_bn)
        context_output = self.dropout(context_relu)
        logits = self.classifier(context_output)
        
        aux_logits = self.aux_branch(context_combined) if self.aux_branch else None
        
        return logits, attn_weights, context_combined, aux_logits

def train_batch(batch_x, batch_y, model, optimizer, criterion, device):
    """
    训练单个batch数据，并返回损失值
    """
    try:
        # 显式指定非阻塞传输
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs, attn_weights, context, aux_logits = model(batch_x)
        loss_main = criterion(outputs, batch_y)
        
        # 动态计算需要加强的类别（当前batch中最少样本的类别）
        class_counts = torch.bincount(batch_y)
        valid_classes = torch.where(class_counts > 0)[0]  # 排除空类别
        aux_class_idx = valid_classes[torch.argmin(class_counts[valid_classes])].item()
        
        aux_mask = (batch_y == aux_class_idx)
        
        if aux_mask.any() and aux_logits is not None:
            loss_aux = criterion(aux_logits[aux_mask], batch_y[aux_mask])
            loss = loss_main + 0.5 * loss_aux
        else:
            loss = loss_main
        
        loss.backward()
        
        # 新增：梯度裁剪，限制梯度范数
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        train_loss = loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train = batch_y.size(0)
        train_correct = (predicted == batch_y).sum().item()
        
        return train_loss, train_correct, total_train
    except Exception as e:
        logging.error(f"Batch训练失败: {str(e)}")
        raise

def train_and_validate(model, train_loader, val_loader, epochs, device, hparams, checkpoint_path=None, resume_epoch=0, resume_optimizer_state=None,
                      writer=None):
    # 在函数开头添加检查
    if writer and not isinstance(writer, SummaryWriter):
        logging.warning("传入的writer参数类型错误，已禁用TensorBoard记录")
        writer = None
    
    model.to(device)
    
    # 动态生成类别权重（基于完整训练集的样本分布）
    # 收集所有训练标签
    all_labels = []
    for _, (_, batch_y) in enumerate(train_loader):
        all_labels.append(batch_y.cpu())
    all_labels = torch.cat(all_labels).numpy()
    
    # 计算类别权重
    unique_classes = np.unique(all_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # 添加类别分布日志
    class_counts = np.bincount(all_labels)
    logging.info("\n=== 训练集类别分布 ===")
    for i, count in enumerate(class_counts):
        logging.info(f"类别 {i}: {count} 样本 ({count/len(all_labels):.2%})")
    logging.info(f"计算得到的类别权重: {class_weights.cpu().numpy().round(2)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    lr = float(hparams.get("lr", 0.001))
    optimizer_type = hparams.get("optimizer", "RMSprop")
    weight_decay = float(hparams.get("weight_decay", 1e-5))
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 新增：设置 warmup scheduler
    warmup_steps = hparams.get("warmup_steps", 100)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    scheduler_warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            logging.info(f"成功恢复优化器状态，从epoch {resume_epoch} 继续训练")
        except Exception as e:
            logging.warning(f"恢复优化器状态失败: {str(e)}")
    
    best_val_acc = 0.0
    patience = hparams.get("patience", 5)
    no_improve = 0

    try:
        start_time = datetime.now()
        for epoch in range(resume_epoch, epochs):
            epoch_start = datetime.now()
            logging.info(f"\n=== 开始 Epoch {epoch+1}/{epochs} ===")
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            total_train = 0
            batch_times = []
            
            # 创建进度条
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f"Epoch {epoch+1}", ncols=100,
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
            try:
                torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化
                torch.cuda.empty_cache()  # 清空GPU缓存
            except:
                pass
            
            for batch_idx, (batch_x, batch_y) in pbar:
                batch_start = datetime.now()
                
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs, attn_weights, context, aux_logits = model(batch_x)
                
                # 主损失计算
                loss_main = criterion(outputs, batch_y)
                
                # 动态计算需要加强的类别（当前batch中最少样本的类别）
                class_counts = torch.bincount(batch_y)
                valid_classes = torch.where(class_counts > 0)[0]  # 排除空类别
                aux_class_idx = valid_classes[torch.argmin(class_counts[valid_classes])].item()
                
                aux_mask = (batch_y == aux_class_idx)
                
                if aux_mask.any() and aux_logits is not None:
                    loss_aux = criterion(aux_logits[aux_mask], batch_y[aux_mask])
                    loss = loss_main + 0.5 * loss_aux
                else:
                    loss = loss_main
                
                loss.backward()
                
                # 新增：梯度裁剪，限制梯度范数
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                optimizer.step()
                scheduler_warmup.step()  # 每个batch更新warmup scheduler
                train_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # 计算并记录batch时间
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_times.append(batch_time)
                
                # 更新进度条描述
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                current_acc = (predicted == batch_y).float().mean().item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_acc:.2%}",
                    'bt(s)': f"{batch_time:.1f}",
                    'lr': f"{lr:.6f}"
                })
                
                # 记录注意力权重直方图和均值（非图片或嵌入可视化，因为数据集非图像）
                if writer and batch_idx == 0:
                    writer.add_histogram('Attention/Weights', attn_weights[0], epoch+1)
                    writer.add_scalar('Attention/MeanWeight', attn_weights[0].mean().item(), epoch+1)
                
                # 添加内存清理
                if batch_idx % 10 == 0:
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
            
            # 关闭进度条
            pbar.close()
            
            avg_train_loss = train_loss / total_train
            train_acc = train_correct / total_train           

            model.eval()
            val_loss = 0.0
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    outputs, _, _, _ = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            avg_val_loss = val_loss / total_val
            val_acc = val_correct / total_val

            # 验证阶段
            val_start = datetime.now()
            val_time = (datetime.now() - val_start).total_seconds()
            
            # 记录epoch统计信息
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            logging.info(f"\n=== Epoch {epoch+1} 统计 ===")
            logging.info(f"训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            logging.info(f"验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.4f}")
            logging.info(f"学习率: {lr:.6f}")
            logging.info(f"Epoch 耗时: {epoch_time//60:.0f}m{epoch_time%60:.2f}s")
            logging.info(f"验证耗时: {val_time:.2f}s")
            logging.info(f"累计运行时间: {(datetime.now()-start_time).total_seconds()//60:.0f}m")
            
            # 更新学习率调度器
            scheduler.step(val_acc)
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"当前学习率: {current_lr:.6f}")
            
            # 保存检查点
            if checkpoint_path is not None:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "hparams": hparams
                }
                
                # 使用临时文件保存，避免中断导致的文件损坏
                temp_checkpoint_path = checkpoint_path + ".tmp"
                torch.save(checkpoint, temp_checkpoint_path)
                os.replace(temp_checkpoint_path, checkpoint_path)
                
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                # 保存最佳模型
                if checkpoint_path is not None:
                    best_model_path = os.path.join(os.path.dirname(checkpoint_path), "best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.info(f"早停：验证准确率在{patience}个epoch内未改善")
                    break
                    
            # 在epoch循环末尾添加TensorBoard记录
            if writer is not None:
                try:
                    writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
                    writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
                    writer.add_scalar('Loss/Val', avg_val_loss, epoch+1)
                    writer.add_scalar('Accuracy/Val', val_acc, epoch+1)
                    writer.add_scalar('Learning Rate', current_lr, epoch+1)
                    
                    # 记录模型参数分布
                    for name, param in model.named_parameters():
                        writer.add_histogram(
                            name, 
                            param.clone().detach().cpu().data.numpy(),  # 添加detach()
                            epoch+1
                        )
                    
                    # 在训练循环中调用时传入model参数
                    log_system_stats(writer, epoch+1, model)
                except Exception as e:
                    logging.warning(f"TensorBoard记录失败: {str(e)}")
                
            # 在每个 batch 输出当前学习率到 TensorBoard
            if writer is not None and batch_idx % 10 == 0:
                current_lr_batch = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning Rate/Batch', current_lr_batch, epoch * len(train_loader) + batch_idx)
                
    except Exception as e:
        logging.error(f"训练过程发生错误: {str(e)}")
        raise
        
    return model

def evaluate_model(model, data_loader, device, writer=None, epoch=0):
    """
    计算模型在给定数据集上的评价指标
    """
    num_classes = model.classifier.out_features
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            outputs, _, _, _ = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())  # 使用extend而不是append
            all_labels.extend(batch_y.cpu().numpy())
    predictions = np.array(all_preds)  # 直接转换list为array
    ground_truth = np.array(all_labels)
    
    # 处理可能的空标签情况
    if len(ground_truth) == 0:
        logging.warning("评估数据集为空!")
        return {
            'accuracy': 0,
            'f1_score': 0,
            'precision': 0,
            'recall': 0,
            'specificity': 0,
            'cohen_kappa': 0
        }
    
    # 修改：在计算分类报告时传入 zero_division 参数，防止计算 precision 时出现警告
    report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(ground_truth, predictions)
    acc = accuracy_score(ground_truth, predictions)
    
    # 修改：为 precision_score, recall_score 和 f1_score 添加 zero_division 参数
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    
    # 新增：计算 Specificity（宏平均）
    specificity_list = []
    for i in range(matrix.shape[0]):
        TP = matrix[i, i]
        FP = matrix[:, i].sum() - TP
        FN = matrix[i, :].sum() - TP
        TN = matrix.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_list.append(spec)
    specificity = np.mean(specificity_list)
    
    # 新增：计算 Cohen's Kappa
    kappa = cohen_kappa_score(ground_truth, predictions)
    
    # 修改：仅输出数字标签对应的分类报告，避免 KeyError
    logging.info("\n=== 分类报告 ===")
    numeric_keys = sorted([key for key in report.keys() if key.isdigit()], key=int)
    for key in numeric_keys:
        logging.info(
            f"类别 {key} - Precision: {report[key]['precision']:.4f} | "
            f"Recall: {report[key]['recall']:.4f} | "
            f"F1: {report[key]['f1-score']:.4f}"
        )
    
    # 输出混淆矩阵
    logging.info("\n=== 混淆矩阵 ===")
    matrix_str = "\n".join([" ".join(map(str, row)) for row in matrix])
    logging.info(f"Confusion Matrix:\n{matrix_str}")
    
    # 添加空预测处理
    if len(np.unique(ground_truth)) < num_classes:
        logging.warning("存在未出现的类别，指标可能不准确")
    
    metrics_dict = {
        'classification_report': report,
        'confusion_matrix': matrix.tolist(),  # 转为 list 便于存储
        'accuracy': acc,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': kappa
    }

    if writer is not None:
        # 动态生成阶段名称
        stage_names = [f'Stage_{i}' for i in range(num_classes)]  # 默认名称
        # 可以从配置中获取实际名称，例如：
        # from sleep_stage_config import STAGE_NAMES
        # stage_names = STAGE_NAMES.get(num_classes, default_stage_names)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay.from_predictions(
            ground_truth,
            predictions,
            display_labels=stage_names,
            ax=ax,
            cmap='Blues',
            colorbar=False
        )
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        plt.tight_layout()
        writer.add_figure('Confusion Matrix', fig, epoch)
    
    return metrics_dict

def run(cfg, base_path, session_id, hparams, args, preloaded_data=None):
    # 将全局随机种子也传入当前超参数中
    hparams["rand_seed"] = args.rand_seed

    # 从 hparams 中获取数据相关参数（否则使用 args 默认值）
    seq_len       = hparams.get("seq_len", args.seq_len)
    modality      = hparams.get("modality", args.modality)
    num_classes   = hparams.get("num_classes", args.num_classes)
    hrv_win_len   = hparams.get("hrv_win_len", args.hrv_win_len)
    batch_size    = hparams.get("batch_size", args.batch_size)
    epochs        = hparams.get("epochs", args.epochs)

    # 分离数据加载：必须传入预加载数据，否则报错
    if preloaded_data is None:
        raise ValueError("必须提供预加载数据, 请先调用 load_data_for_training() 加载数据")
    data = preloaded_data

    (x_train, y_train), (x_val, y_val) = data
    # 确保数据为 NumPy 数组，防止数据类型问题
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)  # 确保数据类型为 float32
    x_val   = torch.tensor(np.array(x_val), dtype=torch.float32)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        y_val   = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)
    else:
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)
        y_val   = torch.tensor(np.array(y_val), dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset   = TensorDataset(x_val, y_val)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 获取特征数量（输入尺寸）以及超参数
    input_size  = x_train.shape[-1]
    lstm_layers = int(hparams["lstm_layers"])
    hidden_size = int(hparams["hidden_size"])
    dropout     = float(hparams.get("dropout", 0.5))
    rnn_type    = str(hparams.get("rnn_type", args.rnn_type))
    num_heads   = int(hparams.get("num_heads", 4))

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionNet(input_size, hidden_size, lstm_layers, num_classes, dropout, rnn_type, num_heads)
    logging.info("模型结构:\n" + str(model))

    # 检查是否存在之前的检查点用于恢复训练
    session_dir = os.path.join(base_path, session_id)
    os.makedirs(session_dir, exist_ok=True)
    checkpoint_path = os.path.join(session_dir, "checkpoint.pth")
    if args.resume:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if checkpoint["hparams"] != hparams:
                    raise ValueError("超参数不匹配，无法恢复训练")
                model.load_state_dict(checkpoint["model_state_dict"])
                start_epoch = checkpoint["epoch"]
                resume_optimizer_state = checkpoint["optimizer_state_dict"]
                logging.info(f"成功从检查点恢复，将从 epoch {start_epoch} 继续训练")
            except Exception as e:
                logging.error(f"检查点加载失败: {str(e)}")
                raise
        else:
            logging.warning("未找到检查点文件，将从头开始训练")
            start_epoch = 0
            resume_optimizer_state = None
    else:
        start_epoch = 0
        resume_optimizer_state = None

    # 修改TensorBoard初始化部分
    writer = None
    if SummaryWriter is not None:
        try:
            tb_base = "/root/tf-logs"
            # 使用参数指定的子目录
            tb_root = os.path.join(tb_base, args.tb_dir)
            os.makedirs(tb_root, exist_ok=True)
        except PermissionError:
            tb_base = os.path.expanduser("~/tf-logs")
            tb_root = os.path.join(tb_base, args.tb_dir)
            logging.warning(f"无权限访问{os.path.join(tb_base, args.tb_dir)}，已切换到用户目录: {tb_root}")
            os.makedirs(tb_root, exist_ok=True, mode=0o755)
        
        time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_dir = os.path.join(tb_root, time_of_run, session_id)
        os.makedirs(tb_dir, exist_ok=True, mode=0o755)
        
        try:
            writer = SummaryWriter(log_dir=tb_dir)
            # 将模型移至指定设备，确保模型参数与输入相同设备
            model.to(device)
            # 添加可视化模型结构图功能：创建虚拟输入，通过 add_graph 记录模型结构
            dummy_input = torch.zeros((1, seq_len, input_size)).to(device)
            writer.add_graph(model, dummy_input)
            logging.info("成功添加模型结构图至TensorBoard")
        except Exception as e:
            logging.error(f"TensorBoard初始化失败或记录模型图失败: {str(e)}")
    else:
        logging.warning("TensorBoard不可用，跳过可视化功能")

    # 修改训练函数调用
    model = train_and_validate(model, train_loader, val_loader, epochs, device, hparams,
                             checkpoint_path=checkpoint_path,
                             resume_epoch=start_epoch,
                             resume_optimizer_state=resume_optimizer_state,
                             writer=writer)  # 传入writer
    
    # 保存模型，文件名包含超参数信息（使用来自 hparams 的数据参数）
    hparams_str = convert_args_to_str(hparams)
    model_file = f"{hparams_str}.pth"
    model_save_path = os.path.join(session_dir, model_file)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"模型已保存到: {model_save_path}")
    
    # 保存评价指标，同时输出Confusion Matrix到TensorBoard（在evaluate_model中添加了writer.add_figure）
    metrics_dict = evaluate_model(model, val_loader, device, writer, start_epoch)
    logging.info("最终验证集评价指标:")
    logging.info(metrics_dict)
    import json
    metrics_file = os.path.join(session_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"评价指标已保存到: {metrics_file}")
    
    # 保存命令行参数文件（便于汇总实验信息）
    args_file = os.path.join(session_dir, "args.txt")
    write_arguments_to_file(vars(args), args_file)
    
    # 在训练结束后添加TensorBoard记录超参数信息
    if writer is not None:
        # 定义指标范围
        metric_dict = {
            'hparam/accuracy': metrics_dict['accuracy'],
            'hparam/f1_score': metrics_dict['f1_score'],
            'hparam/kappa': metrics_dict['cohen_kappa']
        }
        
        # 添加超参数类型定义
        hparam_dict = {
            'lstm_layers': hparams['lstm_layers'],
            'hidden_size': hparams['hidden_size'],
            'lr': f"{hparams['lr']:.2e}",
            'optimizer': hparams['optimizer']
        }
        
        # 动态生成超参数离散域，避免硬编码
        lstm_layers_domain = list(range(1, hparams.get("lstm_layers_max", 4)))  # 默认值：1~3
        optimizer_domain = hparams.get("optimizer_choices", ["Adam", "RMSprop"])
        hparam_domain = {
            "lstm_layers": lstm_layers_domain,
            "optimizer": optimizer_domain
        }
        
        # 根据是否存在tb_run_name决定是否传入run_name参数
        if "tb_run_name" in hparams:
            writer.add_hparams(
                hparam_dict,
                metric_dict,
                run_name=hparams["tb_run_name"],
                hparam_domain_discrete=hparam_domain
            )
        else:
            writer.add_hparams(
                hparam_dict,
                metric_dict,
                hparam_domain_discrete=hparam_domain
            )
            
        # 所有TensorBoard记录完成后关闭writer
        writer.close()
    
    return metrics_dict

def prepare_data(cfg, modality, num_classes, seq_len):
    """
    使用DataLoader加载数据并进行必要的预处理
    """
    data_loader = MyDataLoader(cfg, modality, num_classes, seq_len)
    data_loader.load_windowed_data()
    # 验证数据特征
    x_train, y_train = data_loader.x_train, data_loader.y_train
    x_val, y_val = data_loader.x_val, data_loader.y_val
    
    # 检查数据维度
    logging.info(f"训练数据维度: x_train={x_train.shape}, y_train={y_train.shape}")
    logging.info(f"验证数据维度: x_val={x_val.shape}, y_val={y_val.shape}")
    
    # 检查标签类别数是否匹配
    train_classes = len(np.unique(y_train))
    val_classes = len(np.unique(y_val))
    expected_classes = num_classes  #与当前的num_classes一致 
    
    if train_classes != expected_classes or val_classes != expected_classes:
        raise ValueError(f"标签类别数不匹配! 期望{expected_classes}类, 但训练集有{train_classes}类, 验证集有{val_classes}类")
    
    logging.info(f"数据验证通过: 训练集和验证集的类别数都为{expected_classes}")
    
    return ((x_train, y_train), (x_val, y_val))

def load_data_for_training(cfg, args):
    """
    单独的数据加载函数，用于加载训练和验证数据
    """
    return prepare_data(cfg, args.modality, args.num_classes, args.seq_len)

def run_all_search(cfg, args):
    logging.info("当前脚本路径: " + os.path.abspath(__file__))
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = os.path.join("/root/autodl-tmp/experiment_results", os.path.basename(__file__))
    log_path = os.path.join(log_dir_root, time_of_run)
    os.makedirs(log_path, exist_ok=True)

    setup_logger(log_path)
    # 修改：将参数信息记录到日志中，而不是写入文件
    logging.info("命令行参数: " + str(vars(args)))
    session_index = [0]
    def get_session_id():
        sid = str(session_index[0])
        session_index[0] += 1
        return sid

    # 单独调用数据加载函数，将数据读取和训练搜索分离
    preloaded_data = load_data_for_training(cfg, args)

    try:
        from bayes_opt import BayesianOptimization
    except ImportError:
        logging.error("未检测到 bayes_opt 库, 请安装该库")
        sys.exit(1)

    print("开始训练...")
    def target(lstm_layers, hidden_size, dropout, lr, optimizer_idx, **kwargs):
        optimizer_choices = {0: "RMSprop", 1: "Adam"}
        hparams = {
            "lr": lr,
            "dropout": dropout,
            "optimizer": optimizer_choices[int(round(optimizer_idx))],
            "lstm_layers": int(round(lstm_layers)),
            "hidden_size": int(round(hidden_size)),
            "batch_size": args.batch_size if kwargs.get("batch_size") is None else int(round(kwargs["batch_size"])),
            "epochs": args.epochs if kwargs.get("epochs") is None else int(round(kwargs["epochs"])),
            "seq_len": args.seq_len,
            "modality": args.modality,
            "rnn_type": args.rnn_type,
            "num_classes": args.num_classes,
            "hrv_win_len": args.hrv_win_len,
            "num_heads": int(round(kwargs.get("num_heads", 4))),
            "warmup_steps": int(round(kwargs.get("warmup_steps", 100)))
        }
        session_id = get_session_id()
        logging.info(f"--- Bayesian 搜索训练会话 {session_id} ---")
        print(f"--- Bayesian 搜索训练会话 {session_id} ---")
        print(f"参数信息为：{hparams}")
        try:
            metrics = run(cfg, log_path, session_id, hparams, args, preloaded_data)
            return metrics.get("accuracy", 0)
        except Exception as e:
            logging.error(f"训练过程中发生错误，跳过当前搜索候选。会话 {session_id} 错误信息: {str(e)}")
            return 0

    # 修正参数边界处理逻辑
    pbounds = {}
    if args.batch_size is None:
        pbounds["batch_size"] = (64, 128)
    if args.epochs is None: 
        pbounds["epochs"] = (5, 10)
    pbounds.update({
        "lstm_layers": (1, 3),
        "hidden_size": (64, 256),
        "dropout": (0.3, 0.7),
        "lr": (0.0001, 0.001),
        "optimizer_idx": (0, 1),
        "num_heads": (2, 8),
        "warmup_steps": (50, 200)
    })

    optimizer_BO = BayesianOptimization(
        f=target,
        pbounds=pbounds,
        random_state=args.rand_seed,
        verbose=0
    )
    # 根据命令行参数 --num_total_sessions 设置总实验数
    init_points = 5 if args.num_total_sessions > 5 else args.num_total_sessions
    n_iter = args.num_total_sessions - init_points
    optimizer_BO.maximize(init_points=init_points, n_iter=n_iter)
    
    logging.info("Bayesian Optimization 最佳结果: " + str(optimizer_BO.max))

def summarize_experiments(results_directory):
    """
    优化后的实验结果汇总函数
    """
    import json
    from collections import defaultdict
    
    summary = []
    metrics_summary = defaultdict(list)
    
    try:
        # 递归遍历目录
        for root, dirs, files in os.walk(results_directory):
            if "evaluation_metrics.json" in files and "args.txt" in files:
                try:
                    metrics_file = os.path.join(root, "evaluation_metrics.json")
                    args_file = os.path.join(root, "args.txt")
                    
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                    with open(args_file, "r") as f:
                        args_str = f.read()
                        
                    session = os.path.relpath(root, results_directory)
                    entry = {
                        "session": session,
                        "args": args_str,
                        "metrics": metrics
                    }
                    summary.append(entry)
                    
                    # 收集关键指标
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_summary[metric_name].append(value)
                            
                except Exception as e:
                    logging.warning(f"处理会话 {root} 时出错: {str(e)}")
                    continue
        
        # 打印汇总信息
        print("\n=== 实验结果汇总 ===")
        print(f"总实验次数: {len(summary)}")
        
        # 打印每个指标的统计信息
        print("\n--- 指标统计 ---")
        for metric_name, values in metrics_summary.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                max_val = np.max(values)
                min_val = np.min(values)
                print(f"\n{metric_name}:")
                print(f"  平均值: {mean_val:.4f} ± {std_val:.4f}")
                print(f"  最大值: {max_val:.4f}")
                print(f"  最小值: {min_val:.4f}")
        
        # 打印详细实验信息
        print("\n--- 详细实验信息 ---")
        for entry in summary:
            print(f"\n会话: {entry['session']}")
            print("参数: ", entry["args"])
            print("评价指标: ", entry["metrics"])
            print("-" * 50)
            
        return summary
        
    except Exception as e:
        logging.error(f"汇总实验结果时发生错误: {str(e)}")
        return []

def log_system_stats(writer, epoch, model=None):
    """
    记录系统状态和模型参数统计
    """
    try:
        # 记录 CPU 使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        writer.add_scalar('System/CPU Usage', cpu_percent, epoch)
        
        # 记录内存使用率
        mem = psutil.virtual_memory()
        writer.add_scalar('System/Memory Usage', mem.percent, epoch)
        
        # GPU监控
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                writer.add_scalar(f'System/GPU_{i}_Usage', gpu.load*100, epoch)
                writer.add_scalar(f'System/GPU_{i}_Memory', gpu.memoryUtil*100, epoch)
        except Exception as e:
            logging.debug(f"GPU监控失败: {str(e)}")
            
        # 仅当传入model参数时记录模型参数统计
        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad.detach().cpu().numpy(), epoch)
                writer.add_histogram(f'Weights/{name}', param.detach().cpu().numpy(), epoch)
            
    except Exception as e:
        logging.warning(f"系统监控异常: {str(e)}")

def main():
    args = parse_args()

    # 设置随机种子，保证实验可复现
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        logging.info("使用GPU进行训练")
        torch.cuda.manual_seed_all(args.rand_seed)
    # 新增：设置 cudnn 相关参数，确保可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建配置对象
    cfg = Config()

    logging.info("使用PyTorch进行双向LSTM注意力机制模型超参数调优")
    # 修改：统一调用新的搜索函数，保证不同搜索方式的流程一致
    run_all_search(cfg, args)
    logging.info("调优完成！")


def parse_args():
    """
    定义命令行参数
    """
    parser = argparse.ArgumentParser(description="使用PyTorch实现的双向LSTM注意力机制模型超参数调优")
    parser.add_argument("--num_total_sessions", type=int, default=30, help="大概会创建的会话组数量")
    parser.add_argument("--epochs", type=int, default=5, help="每个试验中的epoch数量")
    parser.add_argument("--batch_size", type=int, default=640, help="训练批大小")
    parser.add_argument("--seq_len", type=int, default=101, help="窗口长度")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--modality", type=str, default="all", help="要使用的模态")
    parser.add_argument("--hrv_win_len", type=int, default=30, help="HRV的时间窗口长度，主要用于文件命名")
    parser.add_argument("--gpu_index", type=int, default=0, help="使用的GPU索引")
    parser.add_argument("--rand_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="RNN类型")
    parser.add_argument("--resume", type=lambda s: s.lower()=='true', default=False, help="是否恢复之前中断的训练, 请传入 'True' 或 'False'")
    parser.add_argument("--tb_dir", type=str, 
                      default=os.path.splitext(os.path.basename(__file__))[0],  # 修改为当前文件名（不带扩展名）
                      help="TensorBoard日志的子目录名称，默认使用当前脚本文件名")
    return parser.parse_args()

if __name__ == '__main__':
    # 仅在没有命令行参数时扩展默认参数，移除 search_method 参数，默认始终使用 Bayesian 搜索
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--epochs", "1000",            
            "--num_total_sessions","100000",
            "--seq_len", "20",  #实际会加1.    传入的 seq_len 是 100、50 或 20，实际在构造样本时，会使用一个长度为 seq_len + 1 的连续数据段，其中前 seq_len 个数据输入模型，最后一个数据作为目标标签。这样的设计可以确保输入数据和对应的标签在时间上的连续性。
            "--batch_size", "1024",   # 1024 是默认的批量大小，如果需要修改，请在命令行中指定 "--batch_size" 参数
            "--modality", "m1",
            "--num_classes","3",
            "--rand_seed", "42",
            "--resume", "False",            
            "--gpu_index","0"
        ])
    main()

    # 调优完成后调用实验汇总函数
    results_directory = os.path.join("/root/autodl-tmp/experiment_results", os.path.basename(__file__))
    summarize_experiments(results_directory)





    