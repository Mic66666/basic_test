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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score
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
import pickle
import traceback

# 添加上级目录到模块搜索路径中（使用相对路径代替硬编码路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from sleep_stage_config import Config
from dataset_builder_loader.data_loader import DataLoader as MyDataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 添加日志功能的设置函数
def setup_logger(log_path):
    # 修改日志路径为当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_dir, exist_ok=True, mode=0o755)
    
    logger = logging.getLogger("SleepStage")
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器保持不变
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 合并为单个文件处理器,记录所有级别日志
    file_handler = logging.FileHandler(
        os.path.join(current_dir, 'sleep_stage.log'),  # 直接使用当前目录
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 记录所有级别日志
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 添加异常处理器
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("未捕获的异常!", exc_info=(exc_type, exc_value, exc_traceback))
        # 对于键盘中断按正常退出处理
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

    sys.excepthook = handle_exception
    
    return logger


class QLearningAgent:
    def __init__(self, hparams_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.hparams_space = hparams_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        
    def generate_random_hparams(self):
        """
        生成一份完整的默认超参数，不遗漏任何key
        """
        full_params = {}
        for key, spec in self.hparams_space.items():
            if isinstance(spec, list):
                full_params[key] = random.choice(spec)
            elif isinstance(spec, tuple):
                low, high, dist_type = spec
                if dist_type == "log-uniform":
                    full_params[key] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                elif dist_type == "int-uniform":  # 新增整数均匀分布处理
                    full_params[key] = random.randint(int(low), int(high))
                else:
                    full_params[key] = np.random.uniform(low, high)
            else:
                # 如遇到无法处理的类型，直接用它本身
                full_params[key] = spec
        return full_params

    def get_action(self, state):
        if state not in self.q_table:
            # 生成合法动作时添加空列表检查
            legal_actions = self.get_legal_actions(state)
            if not legal_actions:
                logging.error(f"状态 {state} 没有合法动作，使用随机超参数")
                legal_actions = [{k: random.choice(v) if isinstance(v, list) else v} 
                               for k, v in self.hparams_space.items()]
            self.q_table[state] = {tuple(a.items()): 0.0 for a in legal_actions}  # 使用元组作为可哈希的键
        
        # 添加空值检查
        if not self.q_table[state]:
            logging.error(f"状态 {state} 的Q表为空，使用随机动作")
            return random.choice(self.get_legal_actions(state))
            
        if random.random() < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def get_legal_actions(self, state):
        try:
            legal_actions = []
            for key, spec in self.hparams_space.items():
                if isinstance(spec, list):
                    legal_actions.extend([{key: v} for v in spec])
                elif isinstance(spec, tuple):
                    low, high, dist_type = spec
                    if dist_type == "log-uniform":
                        action_val = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        action_val = np.random.uniform(low, high)
                    legal_actions.append({key: action_val})
            
            # 确保至少返回一个合法动作
            if not legal_actions:
                logging.warning("未生成合法动作，使用默认超参数空间")
                return [{k: v[0] if isinstance(v, list) else np.random.uniform(*v[:2])} 
                      for k, v in self.hparams_space.items()]
            return legal_actions
        except Exception as e:
            logging.error(f"生成合法动作失败: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")
            # 返回默认动作作为后备
            return [{k: v[0] if isinstance(v, list) else np.random.uniform(*v[:2])} 
                  for k, v in self.hparams_space.items()]
    
    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values()) if next_state in self.q_table else 0.0
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
    
    def search(self, objective_func, num_sessions):
        # 新增默认超参数，用于填补缺失key，避免KeyError
        default_hparams = self.generate_random_hparams()
        best_hparams = None
        best_val_acc = 0.0
        
        for session in range(num_sessions):
            state = str(session)
            done = False
            
            while not done:
                action = self.get_action(state)
                # 将默认超参数与当前动作返回的单一参数合并
                hparams = {**default_hparams, **dict(action)}
                
                val_acc = objective_func(hparams)
                
                if val_acc > best_val_acc:
                    best_hparams = hparams
                    best_val_acc = val_acc
                
                next_state = str(session + 1)
                self.update_q_table(state, action, val_acc, next_state)
                
                state = next_state
                done = session == num_sessions - 1
        
        return best_hparams, best_val_acc

class PPOAgent:
    def __init__(self, hparams_space, policy_lr=3e-4, value_lr=1e-3, train_iters=80, 
                 lam=0.97, target_kl=0.01, clip_ratio=0.2):
        self.hparams_space = hparams_space
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.train_iters = train_iters
        self.lam = lam
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        
        self.policy_model = self.build_policy_model()
        self.value_model = self.build_value_model()
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.policy_lr)
        self.value_optimizer  = torch.optim.Adam(self.value_model.parameters(),  lr=self.value_lr)

    def build_policy_model(self):
        """
        简单Actor网络：将状态转换为离散动作概率分布的logits
        注意：在此示例中，假设状态空间已被简化为某种低维向量，且hparams_space被我们离散化处理。
        若需要更灵活的action表示，需要自行定义映射。
        """
        # 仅示例：三层MLP
        return nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 示例：假设仅输出对若干离散动作的logits(需结合实际动作数)
        )

    def build_value_model(self):
        """
        简单的Value网络：将state映射到一个标量Value
        """
        class ValueNet(nn.Module):
            def __init__(self):
                super(ValueNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, state_tensor):
                return self.net(state_tensor).squeeze(-1)

            def predict(self, states):
                """
                保证与旧接口兼容的API: 输入python数组或list，返回numpy
                """
                self.eval()
                with torch.no_grad():
                    states_t = torch.tensor(states, dtype=torch.float32).unsqueeze(-1)
                    values = self.forward(states_t).cpu().numpy()
                return values
        
        return ValueNet()

    def get_action(self, state):
        # == 简单随机采样改为基于policy_model输出概率分布采样 ==
        try:
            self.policy_model.eval()
            state_t = torch.tensor([state], dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad():
                logits = self.policy_model(state_t)
            # 注意：此处需将logits映射为对离散动作集的分布，示例简化
            # 例如若有N个动作，就输出N维logits, 用torch.distributions.Categorical做采样
            # 这里只做了单输出，演示用
            prob = torch.sigmoid(logits).item()  # [0,1]范围
            # 建立一个离散"动作"的示例
            actions = list(self.get_legal_actions(state))
            if not actions:
                logging.error("PPOAgent.get_action: 未生成合法动作")
                raise ValueError("没有合法动作可供选择")

            # 在此把prob当作二元分布：p选择第0个动作，否则选择第1个
            if random.random() < prob:
                chosen_action = actions[0]
            else:
                chosen_action = actions[min(1, len(actions)-1)]
            return chosen_action
        except Exception as e:
            logging.error(f"动作选择失败: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")
            # 若无可用父类逻辑，可随机返回一个动作，或直接抛出异常
            if actions:
                return random.choice(actions)
            else:
                raise ValueError("没有合法动作可供选择")

    def compute_advantages(self, rewards, values, dones):
        """
        简单GAE实现：advantages = δ + γλδ + ...
        这里以最简形式演示，也可使用更多PPO/GAE超参
        """
        advantages = []
        gae = 0.0
        gamma = 0.99  # 示例常量
        for i in reversed(range(len(rewards))):
            delta = rewards[i] - values[i]
            gae = delta + gamma * self.lam * gae * (1 - dones[i])
            advantages.insert(0, gae)
        return np.array(advantages)

    def update_policy(self, states, actions, old_log_probs, advantages):
        """
        使用PPO的剪切损失(clipped surrogate loss)进行策略更新
        """
        try:
            self.policy_model.train()
            states_t   = torch.tensor(states,   dtype=torch.float32).unsqueeze(-1)
            adv_t      = torch.tensor(advantages, dtype=torch.float32)
            old_logp_t = torch.tensor(old_log_probs, dtype=torch.float32)

            # 假设这里将action进一步映射成索引或one-hot，便于与policy outputs对齐
            # 省略具体动作离散化的细节，以简单logits示例
            # -------------
            lr_clip = self.clip_ratio
            for _ in range(self.train_iters):
                logits = self.policy_model(states_t).squeeze(-1)
                # 假设离散动作(2个动作)，则:
                # distribution = Categorical(logits=some_logits)
                # new_log_probs = distribution.log_prob(action_t)
                # 本示例仅假设action=0/1 => new_log_probs = ...
                new_prob = torch.sigmoid(logits)
                # 动作为0/1 => log_prob(0) = log(1-new_prob)，1 = log(prob)
                # 这里只做非常简化的示例
                chosen_actions = torch.tensor([1 if a else 0 for a in actions], dtype=torch.float32)
                new_log_probs = chosen_actions * torch.log(new_prob + 1e-10) + \
                                (1 - chosen_actions) * torch.log(1 - new_prob + 1e-10)
                
                ratio = torch.exp(new_log_probs - old_logp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - lr_clip, 1.0 + lr_clip) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy     = -(new_prob * torch.log(new_prob+1e-10)
                                + (1-new_prob)*torch.log(1-new_prob+1e-10)).mean()

                loss = policy_loss - 0.01 * entropy  # 加入熵奖励

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
        except Exception as e:
            logging.error(f"策略更新失败: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")

    def update_value(self, states, returns):
        """
        多次迭代，最简MSE损失
        """
        try:
            self.value_model.train()
            states_t = torch.tensor(states,  dtype=torch.float32).unsqueeze(-1)
            returns_t= torch.tensor(returns, dtype=torch.float32)

            for _ in range(self.train_iters):
                pred_values = self.value_model(states_t)
                val_loss = nn.functional.mse_loss(pred_values, returns_t)

                self.value_optimizer.zero_grad()
                val_loss.backward()
                self.value_optimizer.step()

            logging.info("PPOAgent.update_value: 完成价值网络更新")
        except Exception as e:
            logging.error(f"价值网络更新失败: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")

    def search(self, objective_func, num_sessions):
        best_hparams = None
        best_val_acc = 0.0
        
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = 0
        
        for session in range(num_sessions):
            done = False
            while not done:
                # 与 get_action 相同：将state传给policy网络并采样action
                action_dict = self.get_action(state)
                # 这里记录 old_log_prob (示例：以概率进行简单log_prob计算)
                # 省略上下文，实际上需要再次forward计算
                # ...
                
                action = list(action_dict.values())[0]
                hparams = {**action_dict}
                
                val_acc = objective_func(hparams)
                reward = val_acc
                
                if val_acc > best_val_acc:
                    best_hparams = hparams
                    best_val_acc = val_acc
                
                states.append(state)
                actions.append(int(action>0))  # 根据实际action离散化
                rewards.append(reward)
                dones.append(done)
                
                state += 1
                done = session == num_sessions - 1
            
            # 每个session结束后更新模型
            values = self.value_model.predict(states)
            # 由于原始占位中的 policy_model 未实现，此处假定已经在上面存储 log_probs
            # 或者再次前向计算获得
            if not log_probs:
                log_probs = [0.0 for _ in states]  # 示例
            advantages = self.compute_advantages(rewards, values, dones)
            returns = advantages + np.array(values)

            self.update_policy(states, actions, log_probs, advantages)
            self.update_value(states, returns)
            
            # 清空临时记录
            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            log_probs = []
        
        return best_hparams, best_val_acc

def create_agent(method, hparams_space):
    if method == "q_learning":
        return QLearningAgent(hparams_space)
    elif method == "ppo":
        return PPOAgent(hparams_space)
    else:
        logging.error(f"未支持的搜索方法: {method}")
        raise ValueError("未支持的搜索方法")

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

class SleepTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout, num_heads, 
                 use_cbam=True, use_se=True, use_dilated_conv=True, use_short_res_conv=True,
                 use_pre_mhsa=False, use_post_mhsa=False, seq2seq_mode=False):
        super(SleepTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_cbam = use_cbam
        self.use_se = use_se
        self.use_dilated_conv = use_dilated_conv
        self.use_short_res_conv = use_short_res_conv
        self.use_pre_mhsa = use_pre_mhsa
        self.use_post_mhsa = use_post_mhsa
        self.seq2seq_mode = seq2seq_mode
        
        # CNN前端
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 可学习的位置编码
        self.pos_encoder = LearnablePositionalEncoding(hidden_dim)
        
        # Transformer Encoder层
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 前置MHSA
        if use_pre_mhsa:
            self.pre_mhsa = MultiHeadSelfAttention(hidden_dim, num_heads)
        
        # 后置MHSA  
        if use_post_mhsa:
            self.post_mhsa = MultiHeadSelfAttention(hidden_dim, num_heads)
        
        # 特征增强模块
        if use_short_res_conv:
            self.short_res_conv = ShortResConvBlock(hidden_dim)
        if use_dilated_conv:
            self.dilated_conv = DilatedConvBlock(hidden_dim)
        if use_cbam:
            self.cbam = CBAM(hidden_dim)
        if use_se:  
            self.se = SEBlock(hidden_dim)
        
        # 主多分类输出
        if self.seq2seq_mode:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        
        # == 新增(或替换) 多个子任务 (num_classes 个 二分类头) ==
        self.binary_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_classes)
        ])
        
    def forward(self, x):
        # === 新增: 先将维度从 (batch_size, seq_len, input_dim) 调整为 (batch_size, input_dim, seq_len) ===
        x = x.permute(0, 2, 1)

        # CNN前端 (Conv1d 需要 channels 在第二维)
        x = self.cnn(x)

        # 位置编码 (期待输入形状为 [batch, hidden_dim, seq_len])
        x = self.pos_encoder(x)

        # 前置MHSA
        if self.use_pre_mhsa:
            x = self.pre_mhsa(x)

        # === 在进入 PyTorch TransformerEncoder 之前, 需将维度 permute 为 (seq_len, batch, hidden_dim) ===
        x = x.permute(2, 0, 1)  # (batch, hidden_dim, seq_len) -> (seq_len, batch, hidden_dim)
        x = self.transformer_encoder(x)
        # 变回 (batch, hidden_dim, seq_len)
        x = x.permute(1, 2, 0)

        # 后置MHSA
        if self.use_post_mhsa:
            x = self.post_mhsa(x)

        # 特征增强
        if self.use_short_res_conv:
            x = self.short_res_conv(x)
        if self.use_dilated_conv:
            x = self.dilated_conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_se:
            x = self.se(x)
        
        # 分类器
        if self.seq2seq_mode:
            logits = self.classifier(x)
        else:
            logits = self.classifier(x.mean(dim=2))
        
        # == 动态生成多分类logits和 num_classes 个二分类logits ==
        binary_logits_list = []
        pooled_feature = x.mean(dim=2)  # (batch_size, hidden_dim)
        for head in self.binary_heads:
            bin_out = head(pooled_feature)
            binary_logits_list.append(bin_out)

        # 将第二个输出从 list 转为 tuple，避免 JIT trace 的警告
        return logits, tuple(binary_logits_list)

def train_and_validate(
    model, 
    train_loader, 
    val_loader, 
    num_epochs, 
    device, 
    session_dir, 
    hparams, 
    args, 
    start_epoch=0, 
    resume_optimizer_state=None, 
    resume_scheduler_state=None, 
    resume_scaler_state=None,
    writer=None
):
    # 新增：统计总运行时间的起始时刻
    start_time = datetime.now()

    # 改动1: 如果没有传入已有的对象状态，则新建；否则恢复
    if resume_scaler_state is None:
        scaler = torch.amp.GradScaler()
    else:
        scaler = torch.amp.GradScaler()
        scaler.load_state_dict(resume_scaler_state)

    if resume_optimizer_state is None:
        optimizer = optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
        optimizer.load_state_dict(resume_optimizer_state)

    if resume_scheduler_state is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scheduler.load_state_dict(resume_scheduler_state)

    criterion = FocalLoss(gamma=hparams["focal_loss_gamma"])  # 主损失
    
    best_val_acc = 0
    # 新增: 初始化各子任务loss权重（初始值可自行设置），默认为0.2
    subtask_weights = [0.2 for _ in range(model.num_classes)]
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # 改动2: 从 start_epoch 开始计数
    for epoch in range(start_epoch, num_epochs):
        # 新增：统计单个epoch的耗时
        epoch_start = datetime.now()

        # 为后续计算平均batch时间，新增一个列表
        batch_times = []

        # 训练模式
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 统计子任务在训练集的表现
        def init_subtask_metrics(num_classes):
            return {
                "correct": [0]*num_classes,
                "total":   [0]*num_classes
            }
        subtask_train_metrics = init_subtask_metrics(model.num_classes)

        try:
            # 将原先的 "for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), ...):" 修改为：
            pbar = tqdm(
                enumerate(train_loader), 
                total=len(train_loader),
                desc=f"Epoch {epoch+1}", 
                ncols=100, 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            for batch_idx, (inputs, labels) in pbar:
                batch_start = datetime.now()
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 原先使用 torch.cuda.amp.autocast()，现更新为 torch.amp.autocast('cuda')
                with torch.amp.autocast('cuda'):
                    outputs, binary_logits_list = model(inputs)
                    loss_main = criterion(outputs, labels)
                    
                    # 计算多任务总loss，按照 subtask_weights 动态加权
                    total_loss = loss_main
                    for subtask_idx in range(model.num_classes):
                        subtask_labels = (labels == subtask_idx).long()
                        subtask_loss   = criterion(binary_logits_list[subtask_idx], subtask_labels)
                        total_loss    += subtask_weights[subtask_idx] * subtask_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += total_loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total   += labels.size(0)

                # 统计子任务正确率
                for subtask_idx in range(model.num_classes):
                    subtask_pred = binary_logits_list[subtask_idx].argmax(dim=1)
                    subtask_labels = (labels == subtask_idx).long()
                    correct_count = (subtask_pred == subtask_labels).sum().item()
                    subtask_train_metrics["correct"][subtask_idx] += correct_count
                    subtask_train_metrics["total"][subtask_idx]   += subtask_labels.size(0)

                # 计算batch时间，但不再打印
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_times.append(batch_time)

                # 计算最近10个batch的平均处理时间
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0

                current_acc = (predicted == labels).float().mean().item()
                # 在进度条上展示当前loss/acc/batch_time等信息
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'acc': f"{current_acc:.2%}",
                    'bt(s)': f"{batch_time:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # 添加内存清理
                if batch_idx % 10 == 0:
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

        except Exception as e:
            logging.error(f"训练迭代异常: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")
            raise

        train_loss = train_loss / train_total
        train_acc  = train_correct / train_total
        
        # +++ 记录训练指标 +++
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 动态调整子任务权重
        for subtask_idx in range(model.num_classes):
            corr = subtask_train_metrics["correct"][subtask_idx]
            tot  = subtask_train_metrics["total"][subtask_idx]
            subtask_acc = corr / tot if tot > 0 else 0.0

            base_w  = 0.2
            alpha   = 0.2
            new_w   = base_w + alpha*(1.0 - subtask_acc)
            subtask_weights[subtask_idx] = new_w

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        try:
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, binary_logits_list = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total   += labels.size(0)

        except Exception as e:
            logging.error(f"验证迭代异常: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")
            raise
        
        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(session_dir, "best_model.pth"))

        # 记录epoch统计信息
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        logging.info(f"\n=== Epoch {epoch+1} 统计 ===")
        logging.info(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
        logging.info(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        logging.info(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info(f"Epoch 耗时: {epoch_time//60:.0f}m{epoch_time%60:.2f}s")
        logging.info(f"累计运行时间: {(datetime.now()-start_time).total_seconds()//60:.0f}m")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Subtask Weights: {subtask_weights}")
        
        # 每次epoch结束后保存checkpoint
        checkpoint = {
            "epoch": epoch + 1,  # 下次从这个epoch开始
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "hparams": hparams,
            "subtask_weights": subtask_weights
        }
        try:
            torch.save(checkpoint, os.path.join(session_dir, "checkpoint.pth"))
        except Exception as e:
            logging.error(f"检查点保存失败: {str(e)}")
            logging.debug(f"完整堆栈:\n{traceback.format_exc()}")

        # 在训练循环中记录每个epoch的统计信息
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch+1)
            writer.add_scalar('Accuracy/Train', train_acc, epoch+1) 
            writer.add_scalar('Loss/Val', val_loss, epoch+1)
            writer.add_scalar('Accuracy/Val', val_acc, epoch+1)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch+1)

            # 记录模型参数分布
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)

            # 在训练循环中调用时传入model参数
            log_system_stats(writer, epoch+1, model)
            
        # 在每个 batch 输出当前学习率到 TensorBoard
        if writer is not None and batch_idx % 10 == 0:
            current_lr_batch = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate/Batch', current_lr_batch, epoch * len(train_loader) + batch_idx)
            
    return train_losses, train_accs, val_losses, val_accs

def empty_metrics():
    """
    当评估过程出错或数据为空时，返回空的评价指标字典
    """
    return {
        'classification_report': {},
        'confusion_matrix': [],
        'accuracy': 0.0,
        'specificity': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'cohen_kappa': 0.0
    }


class FocalLoss(nn.Module):
    """
    FocalLoss实现，用于处理类别不平衡问题
    """
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码，实现较为简化版本
    """
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, d_model, max_len))
        nn.init.uniform_(self.pe, -0.1, 0.1)

    def forward(self, x):
        # x.shape: (batch, d_model, seq_len)
        seq_len = x.size(2)
        return x + self.pe[:, :, :seq_len]


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块的简单实现
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x):
        # 输入变换: 期望输入形状为 (seq_len, batch, d_model)
        x = x.permute(2, 0, 1)  # (seq_len, batch, d_model)
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.permute(1, 2, 0)  # (batch, d_model, seq_len)
        return attn_output


class ShortResConvBlock(nn.Module):
    """
    短残差卷积块的实现
    """
    def __init__(self, channels):
        super(ShortResConvBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class DilatedConvBlock(nn.Module):
    """
    膨胀卷积块的实现
    """
    def __init__(self, channels, dilation=2):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CBAM(nn.Module):
    """
    CBAM（Convolutional Block Attention Module）的简化实现
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SEBlock(nn.Module):
    """
    SEBlock（Squeeze-and-Excitation）实现
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (batch, channels, seq_len)
        w = x.mean(dim=2)  # 全局平均池化
        w = self.fc(w)
        w = w.unsqueeze(2)
        return x * w

def evaluate_model(model, data_loader, device, writer=None, epoch=0):
    """
    计算模型在给定数据集上的评价指标，新增对各子任务的二分类指标统计(Accuracy、AUC 等)。
    """
    try:
        # 错误修复：直接使用模型自带的num_classes属性
        num_classes = model.num_classes
        model.eval()
        
        # 主多分类结果统计
        all_preds = []
        all_labels = []
        
        # == 新增：子任务的预测和标签，它们是二分类(0或1) ==
        # 以列表套列表的形式存储: subtask_preds[i] 是第i个子任务在整条数据上的预测结果
        subtask_preds  = [[] for _ in range(num_classes)]
        subtask_labels = [[] for _ in range(num_classes)]
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                # 前向传播
                outputs, binary_logits_list = model(batch_x)
                
                # 主多分类预测
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
                # == 新增：记录各子任务的二分类预测/标签 ==
                for i in range(num_classes):
                    # 用 argmax(dim=1) 得到0或1的预测
                    bin_pred = binary_logits_list[i].argmax(dim=1)
                    # 二分类标签：标记batch_y中等于i的为1，否则为0
                    bin_label = (batch_y == i).long()
                    
                    subtask_preds[i].extend(bin_pred.cpu().numpy())
                    subtask_labels[i].extend(bin_label.cpu().numpy())
        
        predictions = np.array(all_preds)
        ground_truth = np.array(all_labels)
        
        # == 新增：如果您有判断空集的逻辑，可以保留 ==
        if len(ground_truth) == 0:
            logging.error("评估数据集为空!")
            return empty_metrics()

        # -----------------------
        # 1) 主多分类评估
        # -----------------------
        class_weights = compute_class_weight('balanced', classes=np.unique(ground_truth), y=ground_truth)
        logging.info(f"类别权重: {class_weights}")
        
        report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
        matrix = confusion_matrix(ground_truth, predictions)
        acc = accuracy_score(ground_truth, predictions)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        specificity_list = []
        for i in range(matrix.shape[0]):
            TP = matrix[i, i]
            FP = matrix[:, i].sum() - TP
            FN = matrix[i, :].sum() - TP
            TN = matrix.sum() - (TP + FP + FN)
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            specificity_list.append(spec)
        specificity = np.mean(specificity_list)
        kappa = cohen_kappa_score(ground_truth, predictions)
        
        logging.info("\n=== 主多分类报告 ===")
        numeric_keys = sorted([key for key in report.keys() if key.isdigit()], key=int)
        for key in numeric_keys:
            logging.info(
                f"类别 {key} - Precision: {report[key]['precision']:.4f} | "
                f"Recall: {report[key]['recall']:.4f} | "
                f"F1: {report[key]['f1-score']:.4f}"
            )
        
        matrix_str = "\n".join([" ".join(map(str, row)) for row in matrix])
        logging.info(f"\n=== 主多分类混淆矩阵 ===\n{matrix_str}")
        
        # -----------------------
        # 2) 子任务(二分类)评估
        # -----------------------
        subtask_metrics = []
        for i in range(num_classes):
            # 转成 numpy array
            st_preds  = np.array(subtask_preds[i])
            st_labels = np.array(subtask_labels[i])
            
            if len(st_labels) == 0:
                logging.warning(f"子任务{i}评估数据为空, 跳过...")
                subtask_metrics.append(None)
                continue
            
            st_acc  = accuracy_score(st_labels, st_preds)
            st_f1   = f1_score(st_labels, st_preds, zero_division=0)
            st_prec = precision_score(st_labels, st_preds, zero_division=0)
            st_rec  = recall_score(st_labels, st_preds, zero_division=0)
            
            # 计算AUC需要预测概率而非argmax，故若需要可以从
            # binary_logits_list[i].softmax(dim=1)[:,1]中获取正类概率(推断阶段即可)
            # 这里只演示accuracy/f1等，不展示prob
            try:
                # 仅示例二分类AUC
                st_auc = roc_auc_score(st_labels, st_preds)
            except ValueError:
                # 若子任务全是同一标签可能无法计算AUC
                st_auc = 0.0

            logging.info(f"[子任务{i}] Accuracy={st_acc:.4f}, Precision={st_prec:.4f}, "
                         f"Recall={st_rec:.4f}, F1={st_f1:.4f}, AUC={st_auc:.4f}")
            
            subtask_metrics.append({
                "accuracy":  st_acc,
                "precision": st_prec,
                "recall":    st_rec,
                "f1_score":  st_f1,
                "auc":       st_auc
            })
        
        # -----------------------
        # 3) 将结果记录到 TensorBoard (若 writer 不为空)
        # -----------------------
        if writer is not None:
            # 主多分类指标
            writer.add_scalar('Eval/Main_Accuracy', acc, epoch)
            writer.add_scalar('Eval/Main_F1', f1, epoch)
            writer.add_scalar('Eval/Main_Kappa', kappa, epoch)
            
            # 各子任务二分类指标
            for i, st_m in enumerate(subtask_metrics):
                if st_m is None:
                    continue
                writer.add_scalar(f'Subtask_{i}/Accuracy',  st_m["accuracy"],  epoch)
                writer.add_scalar(f'Subtask_{i}/Precision', st_m["precision"], epoch)
                writer.add_scalar(f'Subtask_{i}/Recall',    st_m["recall"],    epoch)
                writer.add_scalar(f'Subtask_{i}/F1',        st_m["f1_score"],  epoch)
                writer.add_scalar(f'Subtask_{i}/AUC',       st_m["auc"],       epoch)
            
            # 可视化混淆矩阵等 (主多分类示例)
            stage_names = [f"Stage_{i}" for i in range(num_classes)]
            fig, ax = plt.subplots(figsize=(6, 6))
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
            writer.add_figure('Eval/Main_ConfusionMatrix', fig, epoch)
        
        # -----------------------
        # 4) 打包指标并返回
        # -----------------------
        metrics_dict = {
            'classification_report': report,
            'confusion_matrix': matrix.tolist(),
            'accuracy': acc,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'subtask_metrics': subtask_metrics
        }
        
        return metrics_dict

    except Exception as e:
        logging.error(f"评估过程异常: {str(e)}")
        logging.debug(f"完整堆栈:\n{traceback.format_exc()}")
        return empty_metrics()

def run(cfg, base_path, session_id, hparams, args, preloaded_data=None, session=0):
    # 将全局随机种子也传入当前超参数中
    hparams["rand_seed"] = args.rand_seed
    
    # 创建会话目录
    session_dir = os.path.join(base_path, str(session))
    os.makedirs(session_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(session_dir)
    
    # 记录参数
    write_arguments_to_file(args, os.path.join(session_dir, "args.txt"))
    
    # 加载数据
    if preloaded_data is None:
        data_loader = MyDataLoader(cfg, args.modality, args.num_classes, args.seq_len)
        data_loader.load_windowed_data()
        data = (data_loader.x_train, data_loader.y_train), (data_loader.x_val, data_loader.y_val)
    else:
        data = preloaded_data
    
    # 创建DataLoader
    train_data, val_data = data
    train_x, train_y = train_data
    val_x,   val_y   = val_data
    
    # 将可能是numpy数组或其他类型的数据转成Tensor
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.long)
    val_x   = torch.as_tensor(val_x,   dtype=torch.float32)
    val_y   = torch.as_tensor(val_y,   dtype=torch.long)
    
    # 优先使用命令行参数中的 batch_size，若未指定则使用 hparams 中的 batch_size
    batch_size = args.batch_size if args.batch_size is not None else hparams.get("batch_size", 64)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader   = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False
    )
    
    # 创建模型
    input_size = train_data[0].shape[-1]
    seq_len = train_data[0].shape[-2]
    num_classes = args.num_classes
    
    # 动态调整 num_heads 确保能整除 hidden_dim
    hidden_dim = int(hparams["hidden_dim"])
    original_num_heads = int(hparams.get("num_heads", 4))
    # 找到不大于原始值且能整除 hidden_dim 的最大 num_heads
    adjusted_num_heads = min(original_num_heads, hidden_dim)
    while adjusted_num_heads > 0 and hidden_dim % adjusted_num_heads != 0:
        adjusted_num_heads -= 1
    if adjusted_num_heads == 0:
        adjusted_num_heads = 1  # 至少保留一个注意力头
    
    model = SleepTransformer(
        input_size,
        hidden_dim,
        int(hparams["num_layers"]),
        num_classes,
        float(hparams.get("dropout", 0.5)),
        adjusted_num_heads,
        use_cbam=hparams.get("use_cbam", True),
        use_se=hparams.get("use_se", True),
        use_dilated_conv=hparams.get("use_dilated_conv", True),
        use_short_res_conv=hparams.get("use_short_res_conv", True),
        use_pre_mhsa=hparams.get("use_pre_mhsa", False),
        use_post_mhsa=hparams.get("use_post_mhsa", False),
        seq2seq_mode=hparams.get("seq2seq_mode", False)
    )
    model.num_classes = num_classes  # 添加num_classes属性
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 检查是否需要恢复训练
    start_epoch = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_scaler_state = None
    if args.resume:
        checkpoint_file = os.path.join(session_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            resume_optimizer_state = checkpoint["optimizer_state_dict"]
            resume_scheduler_state = checkpoint["scheduler_state_dict"]
            resume_scaler_state = checkpoint["scaler_state_dict"]
            hparams = checkpoint["hparams"]
            logging.info(f"已从 {checkpoint_file} 恢复训练状态，从第 {start_epoch} 个epoch开始")
        else:
            logging.info(f"未找到检查点文件 {checkpoint_file}，从头开始训练")
    
    # 强制使用命令行参数中的 epochs
    epochs = args.epochs
    
    # 创建SummaryWriter
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
        
        tb_dir = os.path.join(tb_root, session_id, str(session))
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
    
    # === 修改: 将以上提取的状态传给 train_and_validate ===
    train_losses, train_accs, val_losses, val_accs = train_and_validate(
        model, train_loader, val_loader, epochs, device, session_dir, hparams, args,
        start_epoch=start_epoch,
        resume_optimizer_state=resume_optimizer_state,
        resume_scheduler_state=resume_scheduler_state,
        resume_scaler_state=resume_scaler_state,
        writer=writer
    )
    
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
    write_arguments_to_file(args, args_file)
    
    # 关闭TensorBoard SummaryWriter
    if writer is not None:
        writer.close()
    
    return metrics_dict

def prepare_data(cfg, modality, num_classes, seq_len):
    """
    使用DataLoader加载数据并进行必要的预处理
    """
    data_loader = MyDataLoader(cfg, modality, num_classes, seq_len)
    data_loader.load_windowed_data()
    x_train, y_train = data_loader.x_train, data_loader.y_train
    x_val,   y_val   = data_loader.x_val,   data_loader.y_val
    
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
    
    # ========== 新增: 生成每个子任务的 (X, y_subtask) 并进行 1:1 正负样本采样 ==========
    # 示例：对于 num_classes=3 的情况，这里会生成三组 (子任务0, 子任务1, 子任务2) 。
    #       每组是 "i vs not i"的二分类，并进行 1:1 采样。
    
    def make_binary_1to1(x_data, y_data, class_idx):
        """
        将多分类标签转换为 class_idx vs. NOT class_idx 的二分类。
        并进行 1:1 正负样本采样。此函数可以在 3、4、5 甚至更多类别下正常工作。
        """
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        pos_mask = (y_data == class_idx)
        neg_mask = (y_data != class_idx)
        
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        # 如果负样本比正样本多，则下采样负样本
        if len(neg_indices) > len(pos_indices):
            neg_indices = np.random.choice(neg_indices, size=len(pos_indices), replace=False)
        # 如果负样本比正样本少，则上采样负样本
        elif len(neg_indices) < len(pos_indices):
            neg_indices = np.random.choice(neg_indices, size=len(pos_indices), replace=True)
        
        # 最终合并采样后的索引
        new_indices = np.concatenate((pos_indices, neg_indices))
        np.random.shuffle(new_indices)

        # 抽取对应数据
        x_new = x_data[new_indices]
        # 二分类标签：pos=1, neg=0
        y_new = (y_data[new_indices] == class_idx).astype(np.longlong)

        return x_new, y_new

    # 对训练集生成各子任务数据
    binary_tasks_train = []
    for i in range(num_classes):
        x_sub, y_sub = make_binary_1to1(x_train, y_train, i)
        binary_tasks_train.append((x_sub, y_sub))
    
    # 验证集是否也要进行同样的1:1采样取决于实验需求(常规做法保留验证集真实分布)。
    # 若您也需对验证集子任务做1:1采样，可同理调用 make_binary_1to1。
    # 这里示例中保持验证集原分布:
    binary_tasks_val = []
    for i in range(num_classes):
        # 仅做标签转换，不进行1:1采样：
        y_sub = (y_val == i).astype(np.longlong)
        binary_tasks_val.append((x_val, y_sub))

    # 返回原多分类数据 + 每个子任务的二分类数据
    return ((x_train, y_train), (x_val, y_val)), (binary_tasks_train, binary_tasks_val)

def load_data_for_training(cfg, args):
    """
    单独的数据加载函数，用于加载训练和验证数据
    """
    return prepare_data(cfg, args.modality, args.num_classes, args.seq_len)

def run_all_search(cfg, args):
    try:
        # 定义超参数搜索空间
        hparams_space = {
            "learning_rate": (1e-5, 1e-3, "log-uniform"),
            "weight_decay": (1e-6, 1e-3, "log-uniform"),
            "dropout": (0.1, 0.5, "uniform"),
            "hidden_dim": (64, 256, "int-uniform"),  # 改为整数均匀分布
            "num_layers": (2, 6, "int-uniform"),
            "num_heads": (2, 8, "int-uniform"),
            "focal_loss_gamma": (0.5, 5.0, "uniform"),
            "max_grad_norm": (0.5, 5.0, "uniform"),
            "use_cbam": [True, False],
            "use_se": [True, False],
            "use_dilated_conv": [True, False],
            "use_short_res_conv": [True, False],
            "use_pre_mhsa": [True, False],
            "use_post_mhsa": [True, False],
            "seq2seq_mode": [True, False],
        }
        
        # 检查是否存在检查点
        checkpoint_file = os.path.join(r"/root/autodl-tmp/experiment_results", args.tb_dir, "search_checkpoint.pkl")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            agent = checkpoint["agent"]
            start_session = checkpoint["session"] + 1
            best_hparams = checkpoint["best_hparams"]
            best_val_acc = checkpoint["best_val_acc"]
            logging.info(f"从检查点恢复搜索，从第{start_session}个session开始")
        else:
            agent = create_agent(args.search_method, hparams_space)  # 根据指定的搜索方法创建代理
            start_session = 0
            best_hparams = None
            best_val_acc = 0.0
        
        # 运行搜索
        for session in range(start_session, args.num_total_sessions):
            # 生成会话ID
            session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # 创建会话目录
            session_dir = os.path.join(r"/root/autodl-tmp/experiment_results", args.tb_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # 设置日志记录器
            logger = setup_logger(session_dir)
            
            # 记录参数
            write_arguments_to_file(args, os.path.join(session_dir, "args.txt"))
            
            # 加载数据
            data_loader = MyDataLoader(cfg, args.modality, args.num_classes, args.seq_len)
            data_loader.load_windowed_data()
            data = (data_loader.x_train, data_loader.y_train), (data_loader.x_val, data_loader.y_val)
            
            # 定义目标函数
            def objective(hparams, session):
                write_arguments_to_file(hparams, os.path.join(session_dir, "hparams.txt"))
                best_val_acc = run(cfg, session_dir, session_id, hparams, args, data, session)
                return best_val_acc
            
            # 运行一个session
            hparams, val_acc = agent.search(lambda hparams: objective(hparams, session), num_sessions=1)
            
            if val_acc > best_val_acc:
                best_hparams = hparams
                best_val_acc = val_acc
            
            # 保存检查点
            checkpoint = {
                "agent": agent,
                "session": session,
                "best_hparams": best_hparams,
                "best_val_acc": best_val_acc
            }
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)
        
        logging.info(f"最佳超参数: {best_hparams}")
        logging.info(f"最佳验证准确率: {best_val_acc:.4f}")
        
        return best_hparams, best_val_acc
    except Exception as e:
        logging.critical("超参数搜索失败: %s", str(e), exc_info=True)
        raise

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
        logging.debug(f"完整堆栈:\n{traceback.format_exc()}")

def main():
    # 删除临时日志目录设置，直接使用setup_logger中的配置
    setup_logger(None)  # 参数已不再需要
    
    try:
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

        logging.info("使用强化学习进行模型超参数调优")        
        run_all_search(cfg, args)
        logging.info("调优完成！")
    except Exception as e:
        # 移除重复的 error 日志，改为简单退出或仅记录一次
        # 原先是:
        # logging.error("主程序异常: %s", str(e), exc_info=True)
        # sys.exit(1)
        
        # 修改为仅退出(或只简单提示)：
        sys.exit(f"主程序异常: {str(e)}")


def parse_args():
    """
    定义命令行参数
    """
    parser = argparse.ArgumentParser(description="使用强化学习进行超参数调优")
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
    parser.add_argument("--search_method", type=str, default="q_learning", choices=["q_learning", "ppo"], help="搜索方法")
    return parser.parse_args()

if __name__ == '__main__':
    # 初始化应急日志
    emergency_logger = logging.getLogger()
    emergency_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    emergency_logger.addHandler(console_handler)

    try:
        # 仅在没有命令行参数时扩展默认参数，移除 search_method 参数，默认始终使用 Bayesian 搜索
        if len(sys.argv) == 1:
            sys.argv.extend([
                "--epochs", "10000",            
                "--num_total_sessions","1000000",
                "--seq_len", "20",  
                "--batch_size", "1024",
                "--modality", "m1",
                "--num_classes","3",
                "--rand_seed", "42",
                "--resume", "False",            
                "--gpu_index","0",
                "--search_method", "q_learning"
            ])
        main()

        # 调优完成后调用实验汇总函数
        results_directory = os.path.join("/root/autodl-tmp/experiment_results", os.path.basename(__file__))
        summarize_experiments(results_directory)
    except Exception as e:
        emergency_logger.error("致命错误: %s", str(e), exc_info=True)
        sys.exit(1)



    