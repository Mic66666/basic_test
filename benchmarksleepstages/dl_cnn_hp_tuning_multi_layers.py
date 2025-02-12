import argparse
import os
import random
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import logging

# 从已有模块中导入配置与数据加载器
# 添加上级目录到模块搜索路径中
sys.path.insert(0, os.path.abspath(r'/root/ceshi'))
from sleep_stage_config import Config
from dataset_builder_loader.data_loader import DataLoader as MyDataLoader

# 定义命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="使用PyTorch实现的多层CNN超参数调优")
    parser.add_argument("--num_session_groups", type=int, default=30, help="大概会创建的会话组数量")
    parser.add_argument("--epochs", type=int, default=5, help="每个试验中的epoch数量")
    parser.add_argument("--batch_size", type=int, default=640, help="训练批大小")
    parser.add_argument("--nn_type", type=str, default="CNN", help="定义神经网络类型")
    parser.add_argument("--seq_len", type=int, default=100, help="窗口长度")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--modality", type=str, default="all", help="要使用的模态")
    parser.add_argument("--hrv_win_len", type=int, default=30, help="HRV窗口长度")
    parser.add_argument("--gpu_index", type=int, default=0, help="使用的GPU索引")
    parser.add_argument("--rand_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--summary_freq", type=int, default=640, help="记录摘要的频率")
    parser.add_argument("--search_method", type=str, default="grid", choices=["grid", "random"], help="超参数搜索方法")
    return parser.parse_args()

# 辅助函数，将超参数字典转换为字符串（用于保存模型文件名中）
def convert_args_to_str(hparams):
    return "_".join([f"{key}={value}" for key, value in sorted(hparams.items())])

# 将参数写入文件（方便记录当前实验参数）
def write_arguments_to_file(args, file_path):
    with open(file_path, "w") as f:
        f.write(str(args))

# 定义 PyTorch 的多层 CNN 模型（与原始 tf.keras.Sequential 结构对应）
class MultiLayerCNN(nn.Module):
    def __init__(self, input_channels, seq_len, num_conv_layers, conv_filters, kernel_size, num_classes):
        super(MultiLayerCNN, self).__init__()
        layers = []
        # 第一层卷积（输入通道数为 input_channels）
        layers.append(nn.Conv1d(in_channels=input_channels,
                                out_channels=conv_filters,
                                kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2))
        layers.append(nn.ReLU())
        # 后续卷积层均采用相同的超参数
        for _ in range(1, num_conv_layers):
            layers.append(nn.Conv1d(in_channels=conv_filters,
                                    out_channels=conv_filters,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size - 1) // 2))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        # 经过卷积后特征图尺寸为 (conv_filters, seq_len)
        self.fc = nn.Linear(conv_filters * seq_len, num_classes)

    def forward(self, x):
        # x 的形状应为 (batch_size, input_channels, seq_len)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

# 训练和验证过程
def train_and_validate(model, train_loader, val_loader, epochs, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train

        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        avg_val_loss = val_loss / total_val
        val_acc = val_correct / total_val

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, " +
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return model

# run 函数：构造数据 DataLoader，建立模型，训练模型并保存
def run(data, base_path, session_id, hparams, args):
    # data: ((x_train, y_train), (x_val, y_val))
    (x_train, y_train), (x_val, y_val) = data
    # 将 numpy 数组转换为 PyTorch 张量
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_val = torch.tensor(x_val, dtype=torch.float)
    # 如果 y_train 为 one-hot 编码，则转换为类别索引
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        y_val = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)
    else:
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
    # 假设原始数据形状为 (batch_size, seq_len, features), 对于 Conv1d 需要调整为 (batch_size, features, seq_len)
    x_train = x_train.permute(0, 2, 1)
    x_val = x_val.permute(0, 2, 1)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 获取输入特征数量和序列长度
    input_channels = x_train.shape[1]
    seq_len = x_train.shape[2]
    num_conv_layers = hparams["conv_layers"]
    conv_filters = hparams["conv_filter"]
    kernel_size = hparams["conv_len"]
    num_classes = args.num_classes

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    model = MultiLayerCNN(input_channels, seq_len, num_conv_layers, conv_filters, kernel_size, num_classes)
    print(model)

    # 训练和验证
    model = train_and_validate(model, train_loader, val_loader, args.epochs, device)

    # 保存模型（文件名中包含超参数信息）
    hparams_str = convert_args_to_str(hparams)
    model_file = f"model_{args.num_classes}_stages_{args.hrv_win_len}s_{args.nn_type}_{args.seq_len}_seq_{args.modality}_{hparams_str}.pth"
    session_dir = os.path.join(base_path, session_id)
    os.makedirs(session_dir, exist_ok=True)
    model_save_path = os.path.join(session_dir, model_file)
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")

# 调用已有的 DataLoader 加载数据
def prepare_data(cfg, modality, num_classes, seq_len):
    data_loader = MyDataLoader(cfg, modality, num_classes, seq_len)
    data_loader.load_windowed_data()
    return ((data_loader.x_train, data_loader.y_train), (data_loader.x_val, data_loader.y_val))

# 网格搜索：遍历超参数空间的所有组合
def run_all_grid_search(cfg, args):
    print("当前脚本路径:", os.path.abspath(__file__))
    data = prepare_data(cfg, args.modality, args.num_classes, args.seq_len)
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = os.path.join(cfg.CNN_FOLDER, "experiment_results", os.path.basename(__file__))
    log_path = os.path.join(log_dir_root, time_of_run)
    os.makedirs(log_path, exist_ok=True)
    write_arguments_to_file(vars(args), os.path.join(log_path, 'args.txt'))

    session_index = 0
    # 定义各超参数搜索空间
    conv_layers_domain = [1, 2, 3]
    conv_filters_domain = [32, 64, 128]
    filter_len_domain = [3, 5, 7]

    for num_layers in conv_layers_domain:
        for num_filters in conv_filters_domain:
            for filter_len in filter_len_domain:
                hparams = {
                    "conv_layers": num_layers,
                    "conv_filter": num_filters,
                    "conv_len": filter_len
                }
                session_id = str(session_index)
                session_index += 1
                print(f"--- 正在运行训练会话 {session_id} ---")
                print("超参数:", hparams)
                run(data, log_path, session_id, hparams, args)

# 随机搜索：在超参数空间中随机采样
def run_all_random_search(cfg, args):
    print("当前脚本路径:", os.path.abspath(__file__))
    data = prepare_data(cfg, args.modality, args.num_classes, args.seq_len)
    time_of_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_root = os.path.join(cfg.CNN_FOLDER, "experiment_results", os.path.basename(__file__))
    log_path = os.path.join(log_dir_root, time_of_run)
    os.makedirs(log_path, exist_ok=True)
    write_arguments_to_file(vars(args), os.path.join(log_path, 'args.txt'))

    sessions_per_group = 2
    num_sessions = args.num_session_groups * sessions_per_group
    session_index = 0
    conv_layers_domain = [1, 2, 3]
    conv_filters_domain = [32, 64, 128]
    filter_len_domain = [3, 5, 7]
    rng = random.Random(args.rand_seed)

    for _ in range(args.num_session_groups):
        hparams = {
            "conv_layers": rng.choice(conv_layers_domain),
            "conv_filter": rng.choice(conv_filters_domain),
            "conv_len": rng.choice(filter_len_domain)
        }
        for repeat in range(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            print(f"--- 正在运行训练会话 {session_id}/{num_sessions} ---")
            print("超参数:", hparams)
            print("重复次数:", repeat + 1)
            run(data, log_path, session_id, hparams, args)

def main():
    args = parse_args()

    # 设置随机种子，保证实验可复现
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        print("使用GPU进行训练")
        torch.cuda.manual_seed_all(args.rand_seed)

    # 创建配置对象
    cfg = Config()

    print("使用PyTorch进行多层CNN超参数调优")
    if args.search_method == "grid":
        run_all_grid_search(cfg, args)
    else:
        run_all_random_search(cfg, args)
    print("调优完成！")

if __name__ == '__main__':
    # 可在此处直接指定参数，从而无需每次都从命令行传入：
    sys.argv.extend([
        "--epochs", "10", 
        "--batch_size", "128", 
        "--modality", "all",
        "--seq_len", "20"
    ])
    main()