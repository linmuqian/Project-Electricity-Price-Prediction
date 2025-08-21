"""
V1版本: 
1.1 每日特征工程：晚间最大值和均值 (使用了竞价空间和负荷率)
"price",
"日前负荷率(%)",
"日前风电(MW)",
"省调负荷-日前(MW)",
"竞价空间-日前(MW)" 
1.2 并且泄露日前负荷率
"price",
"日前负荷率(%)",
"日前风电(MW)",
1.3: 1.2基础上，序列label和target_fea拼接
1.4: 1.1基础上，序列label和target_fea拼接
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

class MLP(nn.Module):
    """两层MLP模块"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
        self.seq_fea_gru = MLP(
            input_size=seq_feature_size,
            hidden_size=daily_hidden_size,
            output_size=daily_hidden_size,
            dropout=dropout
        )
        
        self.target_fea_gru = MLP(
            input_size=target_feature_size,
            hidden_size=daily_hidden_size,
            output_size=daily_hidden_size,
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
        
    def forward_seq_fea_V1(self, seq_fea: torch.Tensor, seq_label: torch.Tensor) -> torch.Tensor:
        """
        前向传播
    
        Args:
            seq: 输入张量 (batch_size, 7, feature_size)
            label: 输入张量 (batch_size, 7)
        """
        batch_size, num_days, feature_size = seq_fea.size()
        
        # 重塑张量以便处理每天的序列
        seq_fea = seq_fea.view(batch_size * num_days, feature_size)
        
        # 通过每日GRU处理每天的24小时序列
        daily_outputs = self.seq_fea_gru(seq_fea)  # (batch_size * 7, daily_hidden_size)
        
        # 重塑回7天序列
        daily_outputs = daily_outputs.view(batch_size, num_days, -1)

        # hiking label embedding
        label_emb = self.label_tokenizer(seq_label)
        
        # 通过每周GRU处理7天的序列
        weekly_output = self.weekly_gru(daily_outputs+label_emb)  # (batch_size, weekly_hidden_size)
        
        return weekly_output
    
    def forward_seq_fea(self, seq_fea: torch.Tensor, seq_label: torch.Tensor) -> torch.Tensor:
        """
        前向传播
    
        Args:
            seq: 输入张量 (batch_size, 7, feature_size)
            label: 输入张量 (batch_size, 7)
        """
        seq_fea = torch.cat([seq_fea, seq_label.float()[:,:,None]], dim=2)
        batch_size, num_days, feature_size = seq_fea.size()
        
        # 重塑张量以便处理每天的序列
        seq_fea = seq_fea.view(batch_size * num_days, feature_size)
        
        # 通过每日GRU处理每天的24小时序列
        daily_outputs = self.seq_fea_gru(seq_fea)  # (batch_size * 7, daily_hidden_size)
        daily_outputs = daily_outputs.view(batch_size, num_days, -1)
        
        # 通过每周GRU处理7天的序列
        weekly_output = self.weekly_gru(daily_outputs)  # (batch_size, weekly_hidden_size)
        
        return weekly_output
    
    def forward_target_fea(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 24, feature_size)
        """
        batch_size, feature_size = x.size()
        
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