# scripts/train_single_GRU.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')
# 添加项目根目录到路径（加载父目录的包）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.GRU_V1 import HikingPredGRU


# [MOD] 数据集选择器：按 dataset_ver 动态导入 V2 或 V2_4
def get_dataset_class(dataset_ver: str):
    """
    返回对应版本的 HikingDataset 类。
    Args:
        dataset_ver: "V2" 或 "V2_4"
    """
    ver = (dataset_ver or "V2").upper()
    if ver == "V2":
        from utils.hiking_dataset_V2 import HikingDataset
        return HikingDataset
    elif ver == "V2_4":
        from utils.hiking_dataset_V2_4 import HikingDataset
        return HikingDataset
    else:
        raise ValueError(f"Unknown dataset_ver={dataset_ver}. Use 'V2' or 'V2_4'.")


class HikingPredictor: 
    """抬价预测器"""
    
    def __init__(self, 
                 model_config: dict = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化预测器
        
        Args:
            model_config: 模型配置字典
            device: 计算设备
        """
        self.device = device
        
        # 默认配置（保留原有注释）
        default_config = {
            'seq_feature_size': 5,  # 将在训练时自动检测
            'target_feature_size': 4,  # 将在训练时自动检测
            'daily_hidden_size': 32,
            'weekly_hidden_size': 32,
            'mlp_hidden_size': 64,
            'num_classes': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10
        }
        
        self.config = {**default_config, **(model_config or {})}
        
        # 初始化模型
        self.model = HikingPredGRU(**{k: v for k, v in self.config.items() 
                                     if k in ['seq_feature_size','target_feature_size', 'daily_hidden_size', 'weekly_hidden_size', 
                                             'mlp_hidden_size', 'num_classes', 'dropout']})
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train(self, train_loader, val_loader) -> dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史字典
        """
        # 自动检测特征维度（保留原注释）
        # sample_batch = next(iter(train_loader))
        # feature_size = sample_batch[0].shape[-1]  # 获取最后一个维度作为特征维度
        # self.config['feature_size'] = feature_size
        # print(f"自动检测到特征维度: {feature_size}")
        
        # 重新初始化模型
        self.model = HikingPredGRU(**{k: v for k, v in self.config.items() 
                                        if k in ['seq_feature_size', 'target_feature_size', 'daily_hidden_size', 'weekly_hidden_size', 
                                                'mlp_hidden_size', 'num_classes', 'dropout']})
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_seq, batch_target, batch_seq_label, batch_y in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)
                batch_seq_label = batch_seq_label.to(self.device)
                batch_y = batch_y.squeeze().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model((batch_seq, batch_target, batch_seq_label))
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_seq, batch_target, batch_seq_label, batch_y in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_target = batch_target.to(self.device)
                    batch_seq_label = batch_seq_label.to(self.device)
                    batch_y = batch_y.squeeze().to(self.device)
                    
                    outputs = self.model((batch_seq, batch_target, batch_seq_label))
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_hiking_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}]')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
                print('-' * 50)
            
            # 早停
            if patience_counter >= self.config['early_stopping_patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_hiking_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            预测概率和预测标签
        """
        self.model.eval()
        all_probs = []
        all_predictions = []
        
        with torch.no_grad():
            # [MOD] 兼容 3/4 元组；测试集可能是“无标签推理”模式
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 4:
                        batch_seq, batch_target, batch_seq_label, _ = batch
                    elif len(batch) == 3:
                        batch_seq, batch_target, batch_seq_label = batch
                    else:
                        raise ValueError(f"Unexpected batch length: {len(batch)}")
                else:
                    raise ValueError("Unexpected batch type from data_loader.")
                
                batch_seq = batch_seq.to(self.device)
                batch_target = batch_target.to(self.device)
                batch_seq_label = batch_seq_label.to(self.device)
                outputs = self.model((batch_seq, batch_target, batch_seq_label))
                probs = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_probs.append(probs.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        
        # [MOD] 空 loader 友好提示
        if not all_probs:
            raise ValueError(
                "Empty test loader: no inference samples were produced. "
                "Check test date range vs label coverage, sequence_length context, and feature availability."
            )
        
        return np.concatenate(all_probs), np.concatenate(all_predictions)
    
    def plot_training_history(self,save_path):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)


def get_args(version):
    # （保留原有三个版本）
    model_V0_config = {
            'seq_feature_size': 5,  # 将在训练时自动检测
            'target_feature_size': 4,  # 将在训练时自动检测
            'daily_hidden_size': 32,
            'weekly_hidden_size': 32,
            'mlp_hidden_size': 64,
            'num_classes': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 5
        }
    model_V1_config = {
            'seq_feature_size': 11,
            'target_feature_size': 6,
            'daily_hidden_size': 16,
            'weekly_hidden_size': 16,
            'mlp_hidden_size': 32,
            'num_classes': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20,
            'early_stopping_patience': 5
        }
    model_V2_config = {
            'seq_feature_size': 11,
            'target_feature_size': 6,
            'daily_hidden_size': 16,
            'weekly_hidden_size': 16,
            'mlp_hidden_size': 32,
            'num_classes': 2,
            'dropout': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 20,
            'early_stopping_patience': 3
        }

    if version == "V0":
        return model_V0_config
    elif version == "V1":
        return model_V1_config
    elif version == "V2":
        return model_V2_config
    else:
        raise ValueError(f"Invalid version: {version}")


def main(train_begin, train_end, valid_begin, valid_end, test_begin, test_end,
         dataset_ver: str = "V2"):   # [MOD] 增加 dataset_ver 入口参数
    """主函数"""
    # 设置随机种子
    torch.manual_seed(43)
    np.random.seed(42)
    
    model_config = get_args("V2")
    
    # 数据与保存路径（保持原样）
    feature_data_path = 'data/processed/shanxi_new.parquet'
    label_data_path   = 'data/processed/hiking_01_dataset.csv'
    model_save_path   = 'save/curve_classify'
    os.makedirs(model_save_path, exist_ok=True)

    # [MOD] 选择数据集类
    DatasetCls = get_dataset_class(dataset_ver)
    
    # 划分训练/验证/测试数据集
    train_dataset = DatasetCls(
        feature_data_path=feature_data_path,
        label_data_path=label_data_path,
        sequence_length=7,
        feature_dim=24,
        start_time=train_begin,
        end_time=train_end,
        balance_samples=True,
        # [MOD] 训练/验证使用“有标签模式”
        inference_allow_unlabeled=False,
        norm_mu=None, norm_sd=None
    )
    valid_dataset = DatasetCls(
        feature_data_path=feature_data_path,
        label_data_path=label_data_path,
        sequence_length=7,
        feature_dim=24,
        start_time=valid_begin,
        end_time=valid_end,
        balance_samples=False,
        inference_allow_unlabeled=False,
        norm_mu=train_dataset.norm_mu, norm_sd=train_dataset.norm_sd
    )
    test_dataset = DatasetCls(
        feature_data_path=feature_data_path,
        label_data_path=label_data_path,
        sequence_length=7,
        feature_dim=24,
        start_time=test_begin,
        end_time=test_end,
        balance_samples=False,
        # [MOD] 测试允许“无标签目标日”（例如 8/15）进入样本，label=-1
        inference_allow_unlabeled=True,
        norm_mu=train_dataset.norm_mu, norm_sd=train_dataset.norm_sd
    )
    
    # [MOD] 预测前空集检查
    if len(test_dataset) == 0:
        raise RuntimeError(
            f"Test dataset is empty. Check test range [{test_begin}, {test_end}] "
            f"vs label coverage, sequence_length context, and data completeness."
        )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
    
    # 初始化预测器
    predictor = HikingPredictor(model_config, 'cuda')
    
    # 训练模型
    print("开始训练模型...")
    _ = predictor.train(train_loader, valid_loader)
    
    # 可选：绘制训练历史
    predictor.plot_training_history(save_path=os.path.join(model_save_path, 'training_history.png'))
    
    # 验证集评估
    val_probs, val_predictions = predictor.predict(valid_loader)
    val_labels = valid_dataset.labels_dataset
    val_accuracy = 100 * np.mean(np.array(val_predictions) == np.array(val_labels))
    print(f"验证集准确率: {val_accuracy:.2f}%")
    
    # 测试集预测（推理/评估二合一）
    test_probs, test_predictions = predictor.predict(test_loader)
    
    # [MOD] 测试集准确率：仅在存在真标签时计算（跳过 -1）
    test_labels = np.array(test_dataset.labels_dataset)
    mask = test_labels >= 0
    if mask.any():
        test_accuracy = 100 * np.mean(np.array(test_predictions)[mask] == test_labels[mask])
        print(f"测试集准确率(仅含真标签样本): {test_accuracy:.2f}%")
    else:
        print("测试集无真标签（推理模式），仅输出预测与概率。")
    
    # 保存模型
    torch.save(predictor.model.state_dict(), os.path.join(model_save_path, 'hiking_prediction_model.pth'))
    print(f"模型已保存到 {model_save_path}/hiking_prediction_model.pth")
    
    # 输出结果
    predict_df = pd.DataFrame({
        "Date": getattr(test_dataset, "seq_date", []),
        "Pred_Label": np.array(test_predictions),
        "Pred_Probability": np.array(test_probs)[:, 1],
    })
    if mask.any():
        predict_df["True_Label"] = test_labels
    return predict_df


if __name__ == "__main__":
    # [MOD] 可选命令行，方便直接切换 dataset_ver
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_ver", type=str, default="V2", choices=["V2", "V2_4"])
    args = parser.parse_args()

    # demo 入参（保持你原来的默认）
    df = main(
        train_begin="2025-01-08", train_end="2025-04-30",
        valid_begin="2025-05-01", valid_end="2025-05-14",
        test_begin="2025-05-15",  test_end="2025-05-31",
        dataset_ver=args.dataset_ver,
    )
    print(df.head())
