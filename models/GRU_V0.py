"""
V0版本: 
1. 使用每日GRU处理每天24小时序列
2. 使用每周GRU处理7天序列
3. 使用MLP分类器进行分类
4. 使用Embedding进行标签编码
5. 使用Dropout进行正则化
6. 使用ReLU进行激活函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DailyGRU(nn.Module):
    """处理每天24小时序列的GRU模块"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        """
        初始化每日GRU模块
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: dropout率
        """
        super(DailyGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 24, input_size)
            
        Returns:
            输出张量 (batch_size, hidden_size)
        """
        # x shape: (batch_size, 24, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU前向传播
        output, hidden = self.gru(x, h0)
        
        # 返回最后一层的隐藏状态
        return hidden[-1]  # (batch_size, hidden_size)

class WeeklyGRU(nn.Module):
    """处理7天序列的GRU模块"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.1):
        """
        初始化每周GRU模块
        
        Args:
            input_size: 输入特征维度（来自DailyGRU的输出）
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            dropout: dropout率
        """
        super(WeeklyGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 7, input_size)
            
        Returns:
            输出张量 (batch_size, hidden_size)
        """
        # x shape: (batch_size, 7, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # GRU前向传播
        output, hidden = self.gru(x, h0)
        
        # 返回最后一层的隐藏状态
        return hidden[-1]  # (batch_size, hidden_size)

class HikingPredGRU(nn.Module):
    """抬价预测GRU模型"""
    
    def __init__(self, 
                 seq_feature_size: int = None,
                 target_feature_size: int = None,
                 daily_hidden_size: int = 64,
                 weekly_hidden_size: int = 128,
                 mlp_hidden_size: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        """
        初始化抬价预测模型
        
        Args:
            feature_size: 每小时的特征维度
            daily_hidden_size: 每日GRU的隐藏层维度
            weekly_hidden_size: 每周GRU的隐藏层维度
            mlp_hidden_size: MLP隐藏层维度
            num_classes: 分类类别数
            dropout: dropout率
        """
        super(HikingPredGRU, self).__init__()
        
        # 每日GRU模块
        self.seq_fea_gru = DailyGRU(
            input_size=seq_feature_size,
            hidden_size=daily_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        self.target_fea_gru = DailyGRU(
            input_size=target_feature_size,
            hidden_size=daily_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        self.label_tokenizer = nn.Embedding(2, daily_hidden_size)
        
        # 每周GRU模块
        self.weekly_gru = WeeklyGRU(
            input_size=daily_hidden_size,
            hidden_size=weekly_hidden_size,
            num_layers=2,
            dropout=dropout
        )
        
        # MLP分类器
        self.mlp = nn.Sequential(
            nn.Linear(weekly_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size // 2, num_classes)
        )
        
    def forward_seq_fea(self, seq_fea: torch.Tensor, seq_label: torch.Tensor) -> torch.Tensor:
        """
        前向传播
    
        Args:
            seq: 输入张量 (batch_size, 7, 24, feature_size)
            label: 输入张量 (batch_size, 7)
        """
        batch_size, num_days, num_hours, feature_size = seq_fea.size()
        
        # 重塑张量以便处理每天的序列
        seq_fea = seq_fea.view(batch_size * num_days, num_hours, feature_size)
        
        # 通过每日GRU处理每天的24小时序列
        daily_outputs = self.seq_fea_gru(seq_fea)  # (batch_size * 7, daily_hidden_size)
        
        # 重塑回7天序列
        daily_outputs = daily_outputs.view(batch_size, num_days, -1)

        # hiking label embedding
        label_emb = self.label_tokenizer(seq_label)
        
        # 通过每周GRU处理7天的序列
        weekly_output = self.weekly_gru(daily_outputs+label_emb)  # (batch_size, weekly_hidden_size)
        
        return weekly_output
    
    def forward_target_fea(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 24, feature_size)
        """
        batch_size, num_hours, feature_size = x.size()
        
        # 重塑张量以便处理每天的序列
        #x = x.view(batch_size , num_hours, feature_size)
        
        # 通过每日GRU处理每天的24小时序列
        daily_outputs = self.target_fea_gru(x)  # (batch_size * 1, daily_hidden_size)
        return daily_outputs
        
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 7, 24, feature_size)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        seq_fea, target_fea, seq_label = x
        seq_emb = self.forward_seq_fea(seq_fea, seq_label)
        target_emb = self.forward_target_fea(target_fea)
        total_emb = seq_emb + target_emb
        
        # 通过MLP分类器
        output = self.mlp(total_emb)  # (batch_size, num_classes)
        
        return output