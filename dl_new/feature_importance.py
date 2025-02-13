"""
该模块用于计算并判断各个特征在机器学习中的重要性。

参考了 test_2.py 中 prepare_data 的数据处理逻辑，
确保输入数据均为 NumPy 数组，并检查输入维度。
支持多种方法：
    - 使用随机森林分类器计算特征重要性 (method="random_forest")
    - 使用互信息法计算特征重要性 (method="mutual_info")
    - 使用Extra Trees分类器计算特征重要性 (method="extra_trees")
    - 使用置换法计算特征重要性 (method="permutation")
    
该模块也支持独立运行，通过命令行接口进行特征分析。
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import h5py
import os      # 新增：用于目录检查和创建
import sys     # 新增：用于退出程序
import time
import psutil
import re
import traceback

import pandas as pd    # 新增：用于计时
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import rankdata
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

# 新增辅助函数：递归打印H5文件结构
def print_h5_structure(group, indent=0):
    """
    递归打印当前H5文件组的结构
    参数:
        group: h5py.Group 或 h5py.File 对象
        indent: 缩进级别，用于打印结构
    """
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            logging.info("  " * indent + f"Group: {key}")
            print_h5_structure(item, indent + 1)
        else:
            logging.info("  " * indent + f"Dataset: {key}")

def compute_feature_importance(x, y, method="variance", plot=False, feature_names=None, random_state=42, njobs=-1):
    """
    计算输入数据的特征重要性。

    参数:
        x: 输入特征数据 (numpy 数组或可转换为 numpy 数组)，形状应为 (samples, features)
        y: 标签数据 (numpy 数组或可转换为 numpy 数组)
        method: 使用的方法，可选：
            - "variance": 方差阈值
            - "chi2": 卡方统计量（需要非负特征）
            - "extra_trees": Extra Trees
            - "stochastic_lasso": 随机Lasso
            - "forward_floating": 基于 RFECV 的前向浮动搜索
            - "treeshap": 基于 TreeSHAP 的特征重要性
            - "xgboost_hist": 使用直方图优化的 XGBoost
            - "sparse_projection": 稀疏随机投影
            
        plot: 是否绘制重要性条形图，默认为 False
        feature_names: 特征名称列表
        random_state: 随机种子，默认为 42
        njobs: 并行任务数，默认为-1（使用所有核心）

    返回:
        importance_sorted: 按重要性从高到低排序的字典 {特征名称: 重要性分数}
    """
    try:
        # 数据类型和维度检查
        x = np.asarray(x)
        y = np.asarray(y)
        
        if x.ndim != 2:
            raise ValueError(f"输入特征x应为2维数组，当前维度为: {x.ndim}")
        if y.ndim != 1:
            raise ValueError(f"输入标签y应为1维数组，当前维度为: {y.ndim}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"特征样本数({x.shape[0]})与标签样本数({y.shape[0]})不匹配")
            
        # 修改检查缺失值和无穷值的逻辑 - 使用更高效的方法
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("输入数据中存在缺失值(NaN)或无穷值(Inf)")

        # 检查标签值是否合法
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"标签数量过少，至少需要2个不同的类别，当前只有{len(unique_labels)}个类别")
            
        # 新增采样参数
        n_samples = 10000  # 每次采样数量
        n_iterations = 100    # 采样次数
        
        # 检查采样数量是否合理
        if x.shape[0] < n_samples:
            raise ValueError(f"样本数量({x.shape[0]})不足，无法进行{n_samples}条采样")
        
        # 初始化重要性分数存储
        total_importances = np.zeros(x.shape[1])
        valid_iterations = 0
        
        # 添加采样日志头
        logging.info(f"开始进行{n_iterations}次采样计算，每次采样{n_samples}条数据...")

        for iter_idx in range(n_iterations):
            # 生成不同的随机种子确保采样不同
            iter_seed = random_state + iter_idx
            
            try:
                # 修改：为 treeshap 方法使用分层采样，确保每次采样至少包含两个类别
                if method == "treeshap":
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=iter_seed)
                    sample_indices = next(sss.split(x, y))[0]
                    x_sample = x[sample_indices]
                    y_sample = y[sample_indices]
                    logging.info(f"第{iter_idx+1}次采样 (分层采样) - 随机种子为：{iter_seed}，采样后类别分布: {np.unique(y_sample, return_counts=True)}")
                else:
                    np.random.seed(iter_seed)
                    sample_indices = np.random.choice(x.shape[0], n_samples, replace=False)
                    x_sample = x[sample_indices]
                    y_sample = y[sample_indices]
                    logging.info(f"第{iter_idx+1}次采样 - 随机种子为：{iter_seed}，使用样本索引范围: {sample_indices.min()}~{sample_indices.max()}")
                
                # 根据方法计算重要性
                if method == "variance":
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold()
                    selector.fit(x_sample)
                    iter_importances = selector.variances_
                elif method == "chi2":
                    from sklearn.feature_selection import chi2
                    # 确保数据非负
                    x_min = x_sample.min()
                    if x_min < 0:
                        x_transformed = x_sample - x_min
                    else:
                        x_transformed = x_sample
                    scores, _ = chi2(x_transformed, y_sample)
                    iter_importances = scores
                elif method == "extra_trees":
                    from sklearn.ensemble import ExtraTreesClassifier
                    clf = ExtraTreesClassifier(n_estimators=100, random_state=iter_seed, n_jobs=njobs)
                    clf.fit(x_sample, y_sample)
                    iter_importances = clf.feature_importances_
                elif method == "stochastic_lasso":
                    from sklearn.linear_model import LassoCV
                    # 使用 LassoCV 替代 LassoLarsCV，更稳定
                    model = LassoCV(cv=5, random_state=iter_seed)
                    model.fit(x_sample, y_sample)
                    iter_importances = np.abs(model.coef_)
                elif method == "forward_floating":
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.feature_selection import RFECV
                    from sklearn.model_selection import StratifiedKFold
                    
                    # 优化 1: 减少交叉验证折数，从5折改为3折
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=iter_seed)
                    
                    # 优化 2: 使用更轻量级的随机森林配置
                    base_estimator = RandomForestClassifier(
                        n_estimators=100,  # 减少树的数量
                        max_depth=10,     # 限制树的深度
                        min_samples_split=5,
                        random_state=iter_seed,
                        n_jobs=njobs
                    )
                    
                    # 优化 3: 使用更大的步长来减少迭代次数
                    step = max(1, x_sample.shape[1] // 20)  # 每次移除5%的特征
                    
                    # 优化 4: 设置最小特征数，避免过度选择
                    min_features_to_select = max(1, x_sample.shape[1] // 10)
                    
                    selector = RFECV(
                        estimator=base_estimator,
                        step=step,
                        cv=cv,
                        min_features_to_select=min_features_to_select,
                        n_jobs=njobs
                    )
                    
                    # 优化 5: 使用部分样本进行特征选择
                    sample_size = min(5000, x_sample.shape[0])
                    indices = np.random.choice(x_sample.shape[0], sample_size, replace=False)
                    x_subset = x_sample[indices]
                    y_subset = y_sample[indices]
                    
                    selector.fit(x_subset, y_subset)
                    
                    # 优化 6: 使用更高效的重要性分数计算方式
                    ranking = selector.ranking_
                    importance_scores = 1.0 / (ranking + 1e-10)  # 避免除零
                    iter_importances = importance_scores / np.sum(importance_scores)  # 归一化
                elif method == "treeshap":
                    from sklearn.ensemble import RandomForestClassifier
                    import shap
                    
                    # 优化1: 减少用于SHAP计算的样本数量
                    max_shap_samples = x_sample.shape[0]  # 最大样本数
                    if x_sample.shape[0] > max_shap_samples:
                        from sklearn.model_selection import StratifiedShuffleSplit
                        sss = StratifiedShuffleSplit(n_splits=1, train_size=max_shap_samples, random_state=iter_seed)
                        subset_indices = next(sss.split(x_sample, y_sample))[0]
                        x_shap = x_sample[subset_indices]
                        y_shap = y_sample[subset_indices]
                        logging.info(f"SHAP计算使用子样本: {x_shap.shape}")
                    else:
                        x_shap = x_sample
                        y_shap = y_sample
                    
                    # 修复2：添加模型验证
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        random_state=iter_seed,
                        n_jobs=njobs
                    )
                    
                    try:
                        model.fit(x_sample, y_sample)
                        # 计算基础准确率
                        train_acc = model.score(x_shap, y_shap)
                        logging.info(f"基础模型训练准确率: {train_acc:.4f}")
                    except Exception as e:
                        logging.error(f"模型训练失败: {str(e)}")
                        raise RuntimeError(f"随机森林训练失败: {e}")

                    try:
                        explainer = shap.TreeExplainer(
                            model,
                            feature_perturbation="tree_path_dependent",
                            model_output="raw"
                        )
                        
                        # 修复3：添加内存检查
                        mem_info = psutil.virtual_memory()
                        if mem_info.available < x_shap.nbytes * 10:
                            raise MemoryError(f"可用内存不足，需要至少 {x_shap.nbytes * 10} 字节，当前可用 {mem_info.available} 字节")
                            
                        shap_values = explainer.shap_values(x_shap)
                        
                        # 修复4：增强维度验证
                        if isinstance(shap_values, list):
                            # 确保每个类别的SHAP值维度正确
                            for i, class_shap in enumerate(shap_values):
                                if class_shap.ndim != 2 or class_shap.shape[1] != x_sample.shape[1]:
                                    raise ValueError(f"第{i}个类别的SHAP值维度异常，期望(samples, {x_sample.shape[1]})，实际得到{class_shap.shape}")
                            
                            # 合并所有类别的SHAP值（取绝对值后合并）
                            combined_shap = np.concatenate([np.abs(arr) for arr in shap_values], axis=0)
                            iter_importances = np.mean(combined_shap, axis=0)
                        else:
                            # 处理单一类别的情况
                            if shap_values.ndim == 3:
                                # 处理多维SHAP值（例如某些版本返回的形状为(1, samples, features)）
                                combined_shap = np.abs(shap_values).reshape(-1, x_sample.shape[1])
                                iter_importances = np.mean(combined_shap, axis=0)
                            else:
                                if shap_values.shape[1] != x_sample.shape[1]:
                                    raise ValueError(f"SHAP值特征维度不匹配，期望{x_sample.shape[1]}，实际{shap_values.shape[1]}")
                                iter_importances = np.mean(np.abs(shap_values), axis=0)
                        
                        # 添加最终维度验证
                        if iter_importances.shape != (x_sample.shape[1],):
                            raise ValueError(f"最终重要性分数维度错误，期望({x_sample.shape[1]},)，实际得到{iter_importances.shape}")
                            
                    except Exception as e:
                        error_info = f"""
                        SHAP计算失败详情：
                        - 输入形状: x_shap={x_shap.shape}, y_shap={y_shap.shape}
                        - 模型参数: n_estimators={model.n_estimators}, max_depth={model.max_depth}
                        - 内存状态: 已用{mem_info.percent}%，可用{mem_info.available/1024**2:.2f}MB
                        - 错误类型: {type(e).__name__}
                        - 错误信息: {str(e)}
                        """
                        logging.error(error_info)
                        raise RuntimeError("SHAP计算失败，请检查日志") from e
                elif method == "xgboost_hist":
                    import xgboost as xgb
                    model = xgb.XGBClassifier(tree_method='hist', random_state=iter_seed, n_jobs=njobs)
                    model.fit(x_sample, y_sample)
                    iter_importances = model.feature_importances_
                elif method == "sparse_projection":
                    from sklearn.random_projection import SparseRandomProjection
                    from sklearn.metrics import mean_squared_error
                    # 改进稀疏投影方法
                    n_components = min(x_sample.shape[1], 100)  # 限制投影维度
                    projector = SparseRandomProjection(n_components=n_components, random_state=iter_seed)
                    x_projected = projector.fit_transform(x_sample)
                    x_reconstructed = projector.inverse_transform(x_projected)
                    iter_importances = np.array([mean_squared_error(x_sample[:, i], x_reconstructed[:, i])
                                               for i in range(x_sample.shape[1])])
                elif method == "permutation":  # 如果后续添加permutation方法
                    from sklearn.inspection import permutation_importance
                    from sklearn.ensemble import RandomForestClassifier
                    # 创建一个基础模型用于permutation importance
                    estimator = RandomForestClassifier(n_estimators=100, random_state=iter_seed, n_jobs=njobs)
                    estimator.fit(x_sample, y_sample)
                    result = permutation_importance(
                        estimator, x_sample, y_sample, 
                        n_repeats=5, random_state=iter_seed,
                        n_jobs=njobs  # 并行化
                    )
                    iter_importances = result.importances_mean
                else:
                    raise ValueError(f"不支持的方法: {method}")
                
                # 累加重要性分数
                total_importances += iter_importances
                valid_iterations += 1
                logging.info(f"第{iter_idx+1}次采样完成，累计有效迭代次数: {valid_iterations}")

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logging.warning(f"第{iter_idx+1}次采样计算失败。\n错误类型: {type(e).__name__}\n错误信息: {str(e)}\n调用堆栈:\n{error_trace}")
                logging.debug(f"详细错误信息:\n{error_trace}")
                exit()
        
        if valid_iterations == 0:
            raise RuntimeError("所有采样迭代均失败，无法计算特征重要性")
        
        # 计算平均重要性
        importances = total_importances / valid_iterations
        logging.info(f"完成{valid_iterations}次有效采样，平均重要性计算完成")

        # 生成特征名称列表
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(x.shape[1])]
        
        # 优化归一化处理
        if np.all(importances == 0):
            logging.warning(f"{method}方法产生了全零重要性分数，将使用均匀分布")
            importances = np.ones_like(importances) / len(importances)
        else:
            # 使用更稳定的归一化方法
            importances = np.abs(importances)  # 确保非负
            importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances) + 1e-10)
            importances = importances / (np.sum(importances) + 1e-10)

        # 验证结果
        if not np.isclose(np.sum(importances), 1.0, rtol=1e-5):
            raise ValueError(f"归一化后的特征重要性总和为{np.sum(importances)}，应该接近1.0")
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            raise ValueError("特征重要性计算结果包含NaN或Inf值")

        # 生成特征重要性字典，并按重要性降序排序
        importance_dict = dict(zip(feature_names, importances))
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        # 绘制条形图展示特征重要性（如果需要）
        if plot:
            plt.figure(figsize=(12, 8))
            features = list(importance_sorted.keys())
            scores = list(importance_sorted.values())
            
            # 只展示前N个最重要的特征
            n_display = min(20, len(features))
            features = features[:n_display]
            scores = scores[:n_display]
            
            # 使用更美观的配色方案
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(features)))
            bars = plt.bar(range(len(features)), scores, color=colors)
            
            plt.title(f"Top {n_display} Most Important Features ({method})")
            plt.xticks(range(len(features)), features, rotation=45, ha='right')
            plt.xlabel("Feature Names")
            plt.ylabel("Importance Score")
            
            # 优化数值标签格式
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # 使用更安全的文件路径处理
            output_dir = os.path.join("/root/autodl-tmp/experiment_results")
            os.makedirs(output_dir, exist_ok=True)
            safe_method = "".join(c for c in method if c.isalnum() or c in ('_', '-'))
            output_file = os.path.join(output_dir, f"feature_importance_{safe_method}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        logging.info("成功计算特征重要性")
        return importance_sorted
    except Exception as e:
        logging.error(f"使用{method}方法计算特征重要性时发生错误: {str(e)}")
        raise 

# 新增辅助函数：配置日志记录
def setup_logging():
    """
    配置日志记录，将日志同时输出到控制台和文件
    """
    # 设置结果保存目录，并确保目录存在
    results_dir = r"/root/autodl-tmp/experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(os.path.join(results_dir, "feature_importance.log"), mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

# 新增辅助函数：加载和预处理H5文件中的训练数据
def load_training_data(h5_path, needed_features):
    try:
        with h5py.File(h5_path, 'r') as f:
            # 打印整个文件结构
            logging.info("H5文件结构:")
            print_h5_structure(f)
            
            # 获取训练组并检查其存在性
            # 检查所有可用的数据组
            available_groups = list(f.keys())
            for group in available_groups:
                group_obj = f[group]
                logging.info(f"\n组 '{group}' 的详细信息:")
                logging.info(f"- 类型: {type(group_obj).__name__}")
                if isinstance(group_obj, h5py.Group):
                    logging.info(f"- 子项: {list(group_obj.keys())}")
                    for item in group_obj.keys():
                        if isinstance(group_obj[item], h5py.Dataset):
                            logging.info(f"  - 数据集 '{item}' 形状: {group_obj[item].shape}")
                elif isinstance(group_obj, h5py.Dataset):
                    logging.info(f"- 形状: {group_obj.shape}")
                    logging.info(f"- 数据类型: {group_obj.dtype}")
            logging.info(f"可用的数据组: {available_groups}")
            
            # 获取训练组
            train_group = f.get("train")
            if train_group is None:
                logging.error("H5文件中缺少'train'组")
                raise KeyError("H5文件中缺少'train'组")
                
            # 打印训练组的详细信息
            logging.info("\n训练组详细信息:")
            if train_group is not None:
                logging.info(f"训练组键: {list(train_group.keys())}")
                logging.info(f"数据形状:")
                for key in train_group.keys():
                    if isinstance(train_group[key], h5py.Dataset):
                        logging.info(f"- {key}: {train_group[key].shape}")
            
            # 添加数据完整性检查
            if 'block0_values' not in train_group or 'block1_values' not in train_group:
                raise KeyError("缺少必要的数据集")
                
            train_data = train_group['block0_values'][:]
            train_labels = train_group['block1_values'][:, 1]
            
            # 确保特征名称正确解码
            feature_names = []
            for name in train_group['block0_items'][:]:
                if isinstance(name, bytes):
                    try:
                        feature_names.append(name.decode('utf-8'))
                    except UnicodeDecodeError:
                        feature_names.append(name.decode('latin1'))
                else:
                    feature_names.append(str(name))
            
            # 特征过滤和验证
            feature_indices = []
            valid_features = []
            for feat in needed_features:
                try:
                    idx = feature_names.index(feat)
                    feature_indices.append(idx)
                    valid_features.append(feat)
                except ValueError:
                    continue
            
            if not feature_indices:
                raise ValueError("没有找到任何匹配的特征")
            
            # 提取所需特征
            train_data = train_data[:, feature_indices]
            
            
            
            return train_data, train_labels, valid_features
            
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        raise

def analyze_log_file(log_file_path="/root/autodl-tmp/experiment_results/feature_importance.log"):
    """
    分析特征重要性日志文件，统计每个特征的平均重要性分数及总体统计信息，
    同时输出按照平均重要性和平均排名排序的前20位特征。
    
    参数：
        log_file_path: 日志文件路径，默认为 "/root/autodl-tmp/experiment_results/feature_importance.log"
        
    返回：
        一个字典，包含两个键："importance" 和 "rank"，对应的值分别为平均重要性和平均排名字典。
    """
    # 新增：正则表达式，用于匹配特征重要性与平均排名信息
    pattern_feature = re.compile(r"特征\s+(.*?)\s+在各方法下的重要性分数:")
    pattern_avg = re.compile(r"平均重要性分数:\s*([\d\.Ee+-]+)")
    pattern_rank_feature = re.compile(r"特征\s+(.*?)\s+的排名统计:")
    pattern_avg_rank = re.compile(r"平均排名:\s*([\d\.Ee+-]+)")
    
    importance_dict = {}
    rank_dict = {}
    current_importance_feature = None
    current_rank_feature = None
    
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # 同时解析日志中的平均重要性分数和平均排名分数
        for line in lines:
            # 解析平均重要性分数
            feature_match = pattern_feature.search(line)
            if feature_match:
                current_importance_feature = feature_match.group(1).strip()
                continue  # 进入下一行获取平均重要性分数
                
            avg_match = pattern_avg.search(line)
            if avg_match and current_importance_feature:
                try:
                    importance_value = float(avg_match.group(1))
                    importance_dict[current_importance_feature] = importance_value
                except ValueError:
                    print("解析平均重要性分数失败，行内容: " + line)
                current_importance_feature = None
            
            # 解析平均排名分数
            rank_feature_match = pattern_rank_feature.search(line)
            if rank_feature_match:
                current_rank_feature = rank_feature_match.group(1).strip()
                continue  # 进入下一行获取平均排名分数
                
            avg_rank_match = pattern_avg_rank.search(line)
            if avg_rank_match and current_rank_feature:
                try:
                    rank_value = float(avg_rank_match.group(1))
                    rank_dict[current_rank_feature] = rank_value
                except ValueError:
                    print("解析平均排名分数失败，行内容: " + line)
                current_rank_feature = None

        # 分析统计数据 - 平均重要性分数信息
        if not importance_dict:
            print("日志文件中未能解析到任何特征的平均重要性分数信息。")
            return {}
        
        total_features = len(importance_dict)
        all_values = list(importance_dict.values())
        overall_avg = sum(all_values) / total_features
        max_feature = max(importance_dict, key=importance_dict.get)
        min_feature = min(importance_dict, key=importance_dict.get)
        
        print("日志文件分析结果：")
        print(f"特征总数: {total_features}")
        print(f"整体平均重要性分数: {overall_avg:.6f}")
        print(f"最高平均重要性分数: {max_feature} -> {importance_dict[max_feature]:.6f}")
        print(f"最低平均重要性分数: {min_feature} -> {importance_dict[min_feature]:.6f}")
        
        # 按照平均重要性分数降序排序，并输出前100位特征
        sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        print("\n按平均重要性分数排序的前100位特征：")
        for rank, (feature, score) in enumerate(sorted_importance[:100], start=1):
            print(f"{rank}. {feature}: {score:.6f}")
        
        # 按照平均排名分数降序排序（分数越高表示排名越靠前），并输出前100位特征
        if rank_dict:
            sorted_rank = sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)
            print("\n按平均排名分数排序的前100位特征：")
            for rank, (feature, score) in enumerate(sorted_rank[:100], start=1):
                print(f"{rank}. {feature}: {score:.6f}")
        else:
            print("日志文件中未能解析到任何特征的平均排名分数信息。")
        
        # 找出两个列表中的共同特征（前100位特征）
        if rank_dict:
            top_importance_features = [feature for feature, score in sorted_importance[:100]]
            top_rank_features = [feature for feature, score in sorted_rank[:100]]
            common_features = set(top_importance_features) & set(top_rank_features)
            
            print("\n在前100位的平均重要性和前100位的平均排名中共有的特征：")
            if common_features:
                for idx, feature in enumerate(common_features, start=1):
                    print(f"{idx}. {feature}")
            else:
                print("没有共同的特征。")
        else:
            print("\n无法计算共同特征，因为平均排名分数信息缺失。")

        # 打印前10位特征列表（列表形式）
        top10_importance_features = [feature for feature, score in sorted_importance[:10]]
        print("\n【列表形式】前10位平均重要性特征列表:")
        print(top10_importance_features)
        
        if rank_dict:
            top10_rank_features = [feature for feature, score in sorted_rank[:10]]
            print("\n【列表形式】前10位平均排名特征列表:")
            print(top10_rank_features)
        
        # 返回解析到的两种分数信息
        return {"importance": importance_dict, "rank": rank_dict}
    except Exception as e:
        full_trace = traceback.format_exc()
        print(f"分析日志文件时发生错误: {str(e)}")
        print(full_trace)
        raise e

def main():
    setup_logging()
    logging.info("开始特征重要性分析...")

    try:
        # 配置文件路径
        h5_path = r"/root/autodl-tmp/ouputs/HRV30s_ACC30s_H5/hrv30s_acc30s_full_feat_stand.h5"
        feat_list_path = r"/root/ceshi/assets/feature_list.csv"
        
        # 检查文件是否存在
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"找不到H5文件: {h5_path}")
        if not os.path.exists(feat_list_path):
            raise FileNotFoundError(f"找不到特征列表文件: {feat_list_path}")

        # 读取特征列表
        needed_features = pd.read_csv(feat_list_path)['feature_list'].values.tolist()
        logging.info(f"从特征列表中读取了 {len(needed_features)} 个特征")

        # 加载数据
        x, y, feature_names = load_training_data(h5_path, needed_features)
        # 输出数据集信息
        logging.info(f"数据集形状: X={x.shape}, y={y.shape}")
        logging.info(f"特征数量: {len(feature_names)}")
        unique_labels, counts = np.unique(y, return_counts=True)
        logging.info(f"类别分布: {dict(zip(unique_labels, counts))}")
        logging.info(f"样本总数: {len(y)}")
        logging.info(f"特征列表: {feature_names}")

       

        # 定义所有支持的特征重要性计算方法
        methods = [
            "variance",
            "chi2",
            "extra_trees",
            "stochastic_lasso",
            "forward_floating",
            "treeshap",
            "xgboost_hist",
            "sparse_projection",
            "permutation"
        ]

        # 初始化结果存储
        success_methods = []
        all_method_scores = {feature: [] for feature in feature_names}
        all_rank_scores = {feature: [] for feature in feature_names}
        feature_in_top100 = {feature: [] for feature in feature_names}

        # 顺序处理每个方法
        for method in methods:
            start_time = time.time()
            try:
                logging.info(f"正在处理方法: {method}")
                
                # 计算特征重要性
                importance_sorted = compute_feature_importance(
                    x, y,
                    method=method,
                    plot=True,
                    feature_names=feature_names,
                    random_state=42,
                    njobs=-1
                )
                
                # 计算排名
                raw_scores = np.array([importance_sorted.get(f, 0) for f in feature_names])
                ranks = len(raw_scores) - rankdata(raw_scores, method='min') + 1
                rank_scores = dict(zip(feature_names, ranks))
                
                # 获取前100特征
                sorted_features = list(importance_sorted.keys())
                top_n = min(100, len(sorted_features))
                current_top_features = sorted_features[:top_n]
                
                # 更新结果
                success_methods.append(method)
                for feature in feature_names:
                    all_method_scores[feature].append(importance_sorted.get(feature, 0))
                    all_rank_scores[feature].append(rank_scores.get(feature, 0))
                    if feature in current_top_features:
                        feature_in_top100[feature].append(method)
                
                elapsed = time.time() - start_time
                logging.info(f"方法 {method} 成功完成，耗时 {elapsed:.2f} 秒")
                
            except Exception as e:
                logging.error(f"方法 {method} 失败: {str(e)}", exc_info=True)
                elapsed = time.time() - start_time
                logging.info(f"方法 {method} 失败，耗时 {elapsed:.2f} 秒")
                continue

        # ========== 修改：使用实际成功的方法数量进行计算 ==========
        logging.info(f"\n成功完成的方法数量: {len(success_methods)}/{len(methods)}")
        if not success_methods:
            raise ValueError("没有方法成功完成，无法进行结果汇总")

        # 检查有效方法数量
        if len(success_methods) < 2:  # 至少需要3个方法成功
            raise ValueError(f"仅有{len(success_methods)}个方法成功，不足以进行综合评估")

        # 计算所有方法下每个特征的平均重要性（仅统计成功方法）
        average_importance = {}
        for feature, scores in all_method_scores.items():
            if scores:
                # 记录每个特征在各个方法下的具体分数
                logging.info(f"\n特征 {feature} 在各方法下的重要性分数:")
                for method, score in zip(success_methods, scores):
                    logging.info(f"  - {method}: {score:.6f}")
                
                avg_score = sum(scores) / len(scores)
                average_importance[feature] = avg_score
                logging.info(f"  平均重要性分数: {avg_score:.6f}")
            else:
                average_importance[feature] = 0
                logging.info(f"\n特征 {feature} 未能获得有效分数")

        # 根据平均重要性降序排序
        average_importance_sorted = dict(sorted(average_importance.items(), key=lambda item: item[1], reverse=True))
        
        # 记录排序后的综合结果
        logging.info("\n=== 特征重要性综合排名结果 ===")
        logging.info("排名\t特征名称\t平均重要性分数\t标准差\t出现在Top100次数")
        for rank, (feature, avg_score) in enumerate(average_importance_sorted.items(), 1):
            scores = all_method_scores[feature]
            std_dev = np.std(scores) if scores else 0
            top100_count = len(feature_in_top100[feature])
            logging.info(f"{rank}\t{feature}\t{avg_score:.6f}\t{std_dev:.6f}\t{top100_count}/{len(success_methods)}")

        # 计算并记录排名分数
        logging.info("\n=== 特征排名分数统计 ===")
        for feature, rank_scores in all_rank_scores.items():
            if rank_scores:
                avg_rank = sum(rank_scores) / len(rank_scores)
                std_rank = np.std(rank_scores)
                min_rank = min(rank_scores)
                max_rank = max(rank_scores)
                logging.info(f"\n特征 {feature} 的排名统计:")
                logging.info(f"  - 平均排名: {avg_rank:.2f}")
                logging.info(f"  - 排名标准差: {std_rank:.2f}")
                logging.info(f"  - 最佳排名: {min_rank}")
                logging.info(f"  - 最差排名: {max_rank}")
                logging.info(f"  - 各方法具体排名: {dict(zip(success_methods, rank_scores))}")

        # 绘制条形图展示平均特征重要性排序结果
        plt.figure(figsize=(12, 8))
        features = list(average_importance_sorted.keys())
        scores = list(average_importance_sorted.values())

        # 只展示前20个最重要的特征
        if len(features) > 20:
            features = features[:20]
            scores = scores[:20]
            plt.title("Top 20 Most Important Features (Average)")
        else:
            plt.title("Average Feature Importance Ranking")

        bars = plt.bar(range(len(features)), scores, color="skyblue")
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.xlabel("Feature Names")
        plt.ylabel("Average Importance Score")

        # 在柱状图上添加具体数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        output_file = "/root/autodl-tmp/experiment_results/feature_importance_average.png"
        output_directory = os.path.dirname(output_file)
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # ========== 新增平均排名分数计算 ==========
        # 计算所有方法下每个特征的平均排名分数
        average_rank = {}
        for feature, scores in all_rank_scores.items():
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                average_rank[feature] = sum(valid_scores) / len(valid_scores)
            else:
                average_rank[feature] = 0

        # 根据平均排名分数降序排序
        average_rank_sorted = dict(sorted(average_rank.items(), key=lambda item: item[1], reverse=True))
        
        logging.info("\n平均排名分数排序结果（分数越高表示综合排名越靠前）：")
        for feature, avg_score in average_rank_sorted.items():
            logging.info(f"{feature}: {avg_score:.2f} 分")

        # 绘制平均排名分数条形图
        plt.figure(figsize=(12, 8))
        features = list(average_rank_sorted.keys())
        scores = list(average_rank_sorted.values())

        # 只展示前20个最高分特征
        if len(features) > 20:
            features = features[:20]
            scores = scores[:20]
            plt.title("Top 20 Features by Average Rank Score")
        else:
            plt.title("Average Rank Score Ranking")

        bars = plt.bar(range(len(features)), scores, color="salmon")
        plt.xticks(range(len(features)), features, rotation=45, ha='right')
        plt.xlabel("Feature Names")
        plt.ylabel("Average Rank Score")

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        rank_output_file = "/root/autodl-tmp/experiment_results/feature_rank_average.png"
        plt.savefig(rank_output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # ========== 新增：统计成功方法前100特征 ==========
        # 找出在全部成功方法中都进入前100的特征
        common_top_features = []
        for feature, methods_list in feature_in_top100.items():
            # 检查是否存在于所有成功方法的结果中
            if len(methods_list) == len(success_methods):
                common_top_features.append(feature)

        # 记录统计结果
        logging.info("\n===== 成功方法前100特征统计 =====")
        logging.info(f"总成功方法数: {len(success_methods)}")
        logging.info(f"共同进入所有成功方法前100的特征数量: {len(common_top_features)}")
        logging.info("特征列表:")
        for idx, feature in enumerate(common_top_features, 1):
            logging.info(f"{idx}. {feature}")

        # 保存结果到文件
        result_path = "/root/autodl-tmp/experiment_results/common_top100_features.txt"
        with open(result_path, "w") as f:
            f.write("在所有成功方法中均进入前100的重要特征:\n")
            for idx, feature in enumerate(common_top_features, 1):
                f.write(f"{idx}. {feature}\n")

        print(f"\n找到 {len(common_top_features)} 个在所有成功方法中均进入前100的重要特征，结果已保存至 {result_path}")

    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


def compute_feature_union():
    """
    计算三个预定义特征列表的并集
    
    返回:
        list: 包含所有不重复特征的列表
    """
    # 定义三个特征列表
    list1 = ["_Act", "mean_nni", "sdnn", "sdsd", "vlf", "lf", "hf", "lf_hf_ratio", "total_power"]
    
    list2 = ['_anyact_centered_19', 'vlf', '_anyact_19', 'lf_hf_ratio', 'hfnu', 
             '_std_centered_19', 'lfnu', 'hf', '_std_19', 'pnni_20']
    
    list3 = ['_var_centered_1', '_std_centered_1', '_var_1', '_std_1', 'Modified_csi', 
             '_median_1', 'csi', '_min_17', '_mean_2', '_min_centered_10']
    
    # 使用集合运算计算并集
    union_set = set(list1) | set(list2) | set(list3)
    
    # 转换回列表并排序
    union_list = sorted(list(union_set))
    
    # 打印统计信息
    print(f"列表1长度: {len(list1)}")
    print(f"列表2长度: {len(list2)}")
    print(f"列表3长度: {len(list3)}")
    print(f"并集长度: {len(union_list)}")
    
    # 打印重复项统计
    all_items = list1 + list2 + list3
    duplicates = {item: all_items.count(item) for item in set(all_items) if all_items.count(item) > 1}
    if duplicates:
        print("\n重复项统计:")
        for item, count in duplicates.items():
            print(f"'{item}' 出现 {count} 次")
    
    return union_list



if __name__ == "__main__":
    #主函数进行多类别特征分析
    # main() 
    # 分析日志文件
    # analyze_log_file()

    # 测试函数
    result = compute_feature_union()
    print("\n所有特征列表:")
    print(result)
    print(len(result))
   
    