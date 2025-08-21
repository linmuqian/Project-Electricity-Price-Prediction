import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class HikingDataset(Dataset):
    """抬价预测数据集"""
    
    def __init__(self, feature_data_path: str, 
                 label_data_path: str, 
                 sequence_length: int = 7, 
                 feature_dim: int = 24,
                 start_time: str = None,
                 end_time: str = None,
                 balance_samples: bool = False,
                 use_data_augmentation: bool = False):
        """
        初始化数据集
        
        Args:
            feature_data_path: 特征数据文件路径（parquet格式）
            label_data_path: 标签数据文件路径（csv格式）
            sequence_length: 序列长度（天数），仅仅只历史序列，值域0~N
            feature_dim: 每天的特征维度（小时数）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            balance_samples: 是否平衡正负样本比例（默认True）
            use_data_augmentation: 是否使用数据增强技术（默认False）
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.balance_samples = balance_samples
        self.use_data_augmentation = use_data_augmentation
        
        # 读取特征数据
        market_data = pd.read_parquet(feature_data_path, engine='fastparquet')
        market_data.index = pd.to_datetime(market_data.index)
        self.market_data = market_data
        
        # 读取标签数据
        label_data = pd.read_csv(label_data_path)
        label_data['date'] = pd.to_datetime(label_data['date'])
        label_data.set_index('date', inplace=True)
        self.label_data = label_data["da_hiking"].astype(int)
        
        self.market_data = self.market_data.loc[self.label_data.index[0].strftime("%Y-%m-%d"):self.label_data.index[-1].strftime("%Y-%m-%d")]
        
        self.date_index = self.label_data.index
        
        print(f"数据加载完成，共{len(self.label_data)}条记录")
        print(f"特征列数: {len(market_data.columns)}")
        print(f"数据时间范围: {self.label_data.index.min()} 到 {self.label_data.index.max()}")

        # 创建特征（这里用随机特征模拟，实际使用时替换为真实特征）
        self._create_features()
        
        # 创建序列数据
        self._create_sequences()
        
        # 如果指定了时间范围，进行过滤
        if start_time and end_time:
            date_index = (self.seq_date >= pd.to_datetime(start_time)) & (self.seq_date <= pd.to_datetime(end_time))
            self.seq_date = self.seq_date[date_index]
            self.sequences_feature_dataset = self.sequences_feature_dataset[date_index]
            self.sequences_label_dataset = self.sequences_label_dataset[date_index]
            self.target_feature_dataset = self.target_feature_dataset[date_index]
            self.labels_dataset = self.labels_dataset[date_index]
        
        # 如果启用样本平衡，进行正负样本增强
        if self.balance_samples:
            self._balance_samples()
            
        # 如果启用数据增强，进行数据增强
        if self.use_data_augmentation:
            self._enhance_samples()
        
    def _create_features(self):
        """从真实数据创建特征矩阵"""
        price_feature = self.market_data["price"]
        self.price_feature = price_feature.values.reshape(len(self.date_index),24,4).mean(axis=2)
        
        elec_feature_columns = [
            "日前光伏(MW)",
            "日前风电(MW)",
            "省调负荷-日前(MW)"
        ]
        self.elec_feature = self.market_data[elec_feature_columns]
        self.elec_feature["竞价空间-日前(MW)"] = -self.elec_feature["日前光伏(MW)"] - self.elec_feature["日前风电(MW)"] + self.elec_feature["省调负荷-日前(MW)"]
        elec_feature_columns = self.elec_feature.columns.tolist()
        n_elec_features = len(elec_feature_columns)
        self.elec_feature = self.elec_feature.values.reshape(len(self.date_index),24,4,n_elec_features).mean(axis=2)
        
        self.features = np.concatenate([self.price_feature[:,:,None], self.elec_feature], axis=2)
        print("特征矩阵形状:",self.features.shape)
        self.features_mean = self.features.mean(axis=0,keepdims=True).mean(axis=1,keepdims=True)
        self.features_std = self.features.std(axis=0,keepdims=True).std(axis=1,keepdims=True)
        self.features -= self.features_mean
        self.features /= self.features_std
        
    def _create_sequences(self):
        """创建滑动窗口序列"""
        sequences_feature_dataset = []
        sequences_label_dataset = []
        target_feature_dataset = []
        labels_dataset = []
        
        # 获取标签数据
        label_values = self.label_data.values
        seq_date = []
        
        for i in range(len(label_values)):
            if i < self.sequence_length:
                continue
            seq_features = self.features[i-self.sequence_length:i]  # (7, 24, n_features)
            seq_labels = label_values[i-self.sequence_length:i] # (7,)
            target_feature = self.features[i,:,1:]
            target_label = label_values[i] # (1,)
            
            seq_date.append(self.date_index[i])
            sequences_feature_dataset.append(seq_features)
            target_feature_dataset.append(target_feature)
            labels_dataset.append(target_label)
            sequences_label_dataset.append(seq_labels)
            
        self.sequences_feature_dataset = np.array(sequences_feature_dataset)
        self.sequences_label_dataset = np.array(sequences_label_dataset)
        self.target_feature_dataset = np.array(target_feature_dataset)
        self.labels_dataset = np.array(labels_dataset)
        self.seq_date = np.array(seq_date)
        
        print(f"创建了{len(self.sequences_feature_dataset)}个历史序列样本")
        print(f"创建了{len(self.target_feature_dataset)}个目标特征样本")
        print(f"序列形状: {self.sequences_feature_dataset.shape}")
        print(f"目标特征形状: {self.target_feature_dataset.shape}")
        print(f"标签分布: {np.bincount(self.labels_dataset)}")
        
    def _balance_samples(self):
        """平衡正负样本比例，使正负样本比例为1:1"""
        # 获取正负样本索引
        positive_indices = np.where(self.labels_dataset == 1)[0]
        negative_indices = np.where(self.labels_dataset == 0)[0]
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        
        print(f"原始样本分布 - 正样本: {n_positive}, 负样本: {n_negative}")
        
        if n_positive == 0 or n_negative == 0:
            print("警告：只有一种类型的样本，无法进行平衡")
            return
            
        # 确定目标样本数量（取较小的那个数量的2倍，确保1:1比例）
        target_count = min(n_positive, n_negative) * 2
        
        # 对正样本进行过采样（如果正样本数量不足）
        if n_positive < target_count // 2:
            # 计算需要重复的次数
            repeat_times = (target_count // 2) // n_positive
            remainder = (target_count // 2) % n_positive
            
            # 重复正样本
            repeated_positive_indices = np.tile(positive_indices, repeat_times)
            if remainder > 0:
                repeated_positive_indices = np.concatenate([
                    repeated_positive_indices, 
                    np.random.choice(positive_indices, remainder, replace=False)
                ])
            positive_indices = repeated_positive_indices
        
        # 对负样本进行过采样（如果负样本数量不足）
        if n_negative < target_count // 2:
            # 计算需要重复的次数
            repeat_times = (target_count // 2) // n_negative
            remainder = (target_count // 2) % n_negative
            
            # 重复负样本
            repeated_negative_indices = np.tile(negative_indices, repeat_times)
            if remainder > 0:
                repeated_negative_indices = np.concatenate([
                    repeated_negative_indices, 
                    np.random.choice(negative_indices, remainder, replace=False)
                ])
            negative_indices = repeated_negative_indices
        
        # 随机打乱并选择目标数量的样本
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        # 确保正负样本数量相等
        min_count = min(len(positive_indices), len(negative_indices))
        balanced_positive_indices = positive_indices[:min_count]
        balanced_negative_indices = negative_indices[:min_count]
        
        # 合并索引并打乱
        balanced_indices = np.concatenate([balanced_positive_indices, balanced_negative_indices])
        np.random.shuffle(balanced_indices)
        
        # 更新数据集
        self.sequences_feature_dataset = self.sequences_feature_dataset[balanced_indices]
        self.sequences_label_dataset = self.sequences_label_dataset[balanced_indices]
        self.target_feature_dataset = self.target_feature_dataset[balanced_indices]
        self.labels_dataset = self.labels_dataset[balanced_indices]
        self.seq_date = [self.seq_date[i] for i in balanced_indices]
        
        print(f"样本平衡后 - 总样本数: {len(self.labels_dataset)}")
        print(f"平衡后标签分布: {np.bincount(self.labels_dataset)}")
        print(f"正负样本比例: {np.bincount(self.labels_dataset)[1]}:{np.bincount(self.labels_dataset)[0]}")
        
    def _enhance_samples(self, noise_std=0.01, time_shift_range=1):
        """
        使用数据增强技术增强正样本
        
        Args:
            noise_std: 添加噪声的标准差
            time_shift_range: 时间偏移范围（小时）
        """
        # 获取正样本索引
        positive_indices = np.where(self.labels_dataset == 1)[0]
        
        if len(positive_indices) == 0:
            print("没有正样本可以增强")
            return
            
        # 创建增强后的样本
        enhanced_sequences = []
        enhanced_target_features = []
        enhanced_labels = []
        enhanced_seq_labels = []
        enhanced_dates = []
        
        for idx in positive_indices:
            # 原始样本
            seq_feature = self.sequences_feature_dataset[idx].copy()
            target_feature = self.target_feature_dataset[idx].copy()
            seq_label = self.sequences_label_dataset[idx].copy()
            label = self.labels_dataset[idx]
            date = self.seq_date[idx]
            
            # 1. 添加高斯噪声
            noise = np.random.normal(0, noise_std, seq_feature.shape)
            enhanced_seq_feature = seq_feature + noise
            
            target_noise = np.random.normal(0, noise_std, target_feature.shape)
            enhanced_target_feature = target_feature + target_noise
            
            enhanced_sequences.append(enhanced_seq_feature)
            enhanced_target_features.append(enhanced_target_feature)
            enhanced_labels.append(label)
            enhanced_seq_labels.append(seq_label)
            enhanced_dates.append(date)
            
            # 2. 时间偏移增强（随机选择1-2个样本）
            if np.random.random() < 0.5:
                # 随机时间偏移
                shift = np.random.randint(-time_shift_range, time_shift_range + 1)
                if shift != 0:
                    # 对序列特征进行时间偏移
                    shifted_seq_feature = np.roll(seq_feature, shift, axis=1)
                    shifted_target_feature = np.roll(target_feature, shift, axis=0)
                    
                    enhanced_sequences.append(shifted_seq_feature)
                    enhanced_target_features.append(shifted_target_feature)
                    enhanced_labels.append(label)
                    enhanced_seq_labels.append(seq_label)
                    enhanced_dates.append(date)
        
        # 将增强的样本添加到原始数据集
        if enhanced_sequences:
            self.sequences_feature_dataset = np.concatenate([
                self.sequences_feature_dataset, 
                np.array(enhanced_sequences)
            ], axis=0)
            
            self.target_feature_dataset = np.concatenate([
                self.target_feature_dataset, 
                np.array(enhanced_target_features)
            ], axis=0)
            
            self.labels_dataset = np.concatenate([
                self.labels_dataset, 
                np.array(enhanced_labels)
            ], axis=0)
            
            self.sequences_label_dataset = np.concatenate([
                self.sequences_label_dataset, 
                np.array(enhanced_seq_labels)
            ], axis=0)
            
            self.seq_date.extend(enhanced_dates)
            
            print(f"数据增强后 - 总样本数: {len(self.labels_dataset)}")
            print(f"增强后标签分布: {np.bincount(self.labels_dataset)}")
        
    def __len__(self):
        return len(self.labels_dataset)
    
    def __getitem__(self, idx):
        return_data = (
            torch.FloatTensor(self.sequences_feature_dataset[idx]),
            torch.FloatTensor(self.target_feature_dataset[idx]),
            torch.LongTensor(self.sequences_label_dataset[idx]),
            torch.LongTensor([self.labels_dataset[idx]]),
        )
        return return_data


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例，启用样本平衡和数据增强
    dataset = HikingDataset(
        feature_data_path="data/processed/shanxi_new.parquet",
        label_data_path="data/processed/hiking_01_dataset.csv",
        sequence_length=7,
        start_time="2025-01-01",
        end_time="2025-01-31",
        balance_samples=True,  # 启用样本平衡
        use_data_augmentation=False  # 启用数据增强
    )
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 检查样本分布
    labels = dataset.labels_dataset
    print(f"最终样本分布: {np.bincount(labels)}")
    print(f"正负样本比例: {np.bincount(labels)[1]}:{np.bincount(labels)[0]}")
    
    # 获取一个批次的数据
    for batch in dataloader:
        sequences, target_features, seq_labels, labels = batch
        print(f"批次形状:")
        print(f"  序列特征: {sequences.shape}")
        print(f"  目标特征: {target_features.shape}")
        print(f"  序列标签: {seq_labels.shape}")
        print(f"  标签: {labels.shape}")
        break