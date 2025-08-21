# hiking_dataset_V2.py  —— V2 + 推理开关（inference_allow_unlabeled）
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class HikingDataset(Dataset):
    """抬价预测数据集（V2_4：省调/竞价夜间max/mean替换为“当日−前D个hiking=0日均值”差分）
       新增：inference_allow_unlabeled=True 时，允许无标签目标日（如 8/15）进入样本，label 用 -1 占位
    """
    def __init__(self, feature_data_path: str, 
                 label_data_path: str, 
                 sequence_length: int = 7, 
                 feature_dim: int = 24,
                 start_time: str = None,
                 end_time: str = None,
                 balance_samples: bool = False,
                 use_data_augmentation: bool = False,
                 inference_allow_unlabeled: bool = False,
                 norm_mu: np.ndarray = None,
                 nrom_sd: np.ndarray = None):   
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.balance_samples = balance_samples
        self.use_data_augmentation = use_data_augmentation
        self.inference_allow_unlabeled = inference_allow_unlabeled
        
        # 读取特征数据（15min 索引）
        market_data = pd.read_parquet(feature_data_path, engine='fastparquet')
        market_data.index = pd.to_datetime(market_data.index)
        self.market_data = market_data
        
        # 读取标签数据（天级）
        label_data = pd.read_csv(label_data_path)
        label_data['date'] = pd.to_datetime(label_data['date'])
        label_data.set_index('date', inplace=True)
        # self.label_data = label_data["da_hiking"].astype(int)  # 原：整型不支持 NaN
        self.label_data = pd.to_numeric(label_data["da_hiking"], errors="coerce")  # [MOD] 允许 NaN，便于推理
        
        # 日期轴：训练/验证裁到标签范围；推理使用特征的自然日期轴（包含未来）
        if not self.inference_allow_unlabeled:
            self.market_data = self.market_data.loc[
                self.label_data.index[0].strftime("%Y-%m-%d"):
                self.label_data.index[-1].strftime("%Y-%m-%d")
            ]
            self.date_index = self.label_data.index
        else:
            self.date_index = pd.DatetimeIndex(sorted(set(self.market_data.index.normalize())))
        
        print(f"数据加载完成，共{len(self.label_data)}条记录")
        print(f"特征列数: {len(market_data.columns)}")
        print(f"数据时间范围(标签): {self.label_data.index.min()} 到 {self.label_data.index.max()}")

        # 1) 构造特征
        self._create_features()
        # 2) 组装序列样本
        self._create_sequences()
        
        # 可选时间窗过滤
        if start_time and end_time:
            date_mask = (self.seq_date >= pd.to_datetime(start_time)) & (self.seq_date <= pd.to_datetime(end_time))
            self.seq_date = self.seq_date[date_mask]
            self.sequences_feature_dataset = self.sequences_feature_dataset[date_mask]
            self.sequences_label_dataset = self.sequences_label_dataset[date_mask]
            self.target_feature_dataset = self.target_feature_dataset[date_mask]
            self.labels_dataset = self.labels_dataset[date_mask]
        
        # 样本平衡 / 数据增强：仅训练/验证执行
        if (not self.inference_allow_unlabeled) and self.balance_samples:
            self._balance_samples()
        if (not self.inference_allow_unlabeled) and self.use_data_augmentation:
            self._enhance_samples()
        
    def _create_features(self):
        """
        从真实数据创建特征矩阵：
        - 基础10维：夜间max(5列) + 夜间mean(5列)
        - 其中【省调负荷-日前】【竞价空间-日前】两列的夜间 max/mean 替换为：
          当日统计量 − 前D个（hiking=0）日该统计量的均值（不含当天）
        """
        # 使用的原始列（顺序固定）
        elec_feature_columns = [
            "price",                # 0
            "日前负荷率(%)",         # 1
            "日前风电(MW)",          # 2
            "省调负荷-日前(MW)",      # 3  ← 差分(hiking=0)
            "竞价空间-日前(MW)"       # 4  ← 差分(hiking=0)
        ]
        self.elec_feature = self.market_data[elec_feature_columns].copy()

        # 每日 96 条校验
        if len(self.elec_feature) % 96 != 0:
            raise ValueError("原始特征行数必须是96的整数倍（每日96条15min数据）")

        n_elec_features = len(elec_feature_columns)
        D = 4           # 参考最近 D 个 hiking=0 日
        STEPS = 96      # 每天 96 个点（15min）
        NIGHT_BEG = 64  # 夜间窗口起点（16:00），窗口为 [64:96)
        
        # 形状：(N_day, 96, n_elec)
        X = self.elec_feature.values.reshape(-1, STEPS, n_elec_features)
        night = X[:, NIGHT_BEG:, :]  # 夜间窗口 16:00-24:00
        
        # 日级统计（NaN 安全）
        day_max  = np.nanmax(night, axis=1)   # (N_day, n_elec)
        day_mean = np.nanmean(night, axis=1)

        # [MOD] 对整列 NaN 做前后填充
        for arr in (day_max, day_mean):
            for j in range(arr.shape[1]):
                s = pd.Series(arr[:, j])
                if s.isna().any():
                    arr[:, j] = s.ffill().bfill().fillna(0.0).to_numpy()

        # 标签（对齐到 date_index；推理模式下可能包含 NaN）
        labels_by_day = self.label_data.reindex(self.date_index).to_numpy()  # (N_day,)

        def _diff_vs_prevD_h0(stat_1d: np.ndarray, labels: np.ndarray, d: int = D) -> np.ndarray:
            """
            y[t] = stat_1d[t] - mean( stat_1d[idx] ), idx 为 t 之前且 labels[idx]==0 的最近 d 个。
            若取不到历史，则置 0（等价于用自身作均值）。
            """
            N = len(stat_1d)
            out = np.zeros(N, dtype=float)
            for t in range(N):
                prev_idx = np.where(labels[:t] == 0)[0]
                if prev_idx.size == 0:
                    out[t] = 0.0
                else:
                    take = prev_idx[-d:]
                    out[t] = stat_1d[t] - np.nanmean(stat_1d[take])
            return out

        # 对列 3、4（省调/竞价）的 max/mean 做差分替换
        for j in (3, 4):
            day_max[:,  j] = _diff_vs_prevD_h0(day_max[:,  j], labels_by_day, D)
            day_mean[:, j] = _diff_vs_prevD_h0(day_mean[:, j], labels_by_day, D)
        
        # 拼成最终日特征：(N_day, 10) = [max(5列) || mean(5列)]
        self.elec_feature = np.concatenate([day_max, day_mean], axis=1)
        
        # 目标日不可用的信息：屏蔽 price / 负荷率 的 max 和 mean
        # 顺序：[max:0..4] + [mean:5..9] → 屏蔽索引 [0,1,5,6]
        self.mask_index = np.ones(self.elec_feature.shape[1], dtype=bool)
        self.mask_index[[0, 1, 5, 6]] = False
        
        # [MOD] 无泄露标准化（逐列）+ 清理非有限数
        f = self.elec_feature.astype(float)
        mu = self.norm_mu if self.norm_mu is not None else np.nanmean(f, axis=0, keepdims=True)
        sd = self.norm_sd if self.norm_sd is not None else np.nanstd( f, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        f = (f - mu) / sd
        f[~np.isfinite(f)] = 0.0
        self.features = f
        self.norm_mu, self.norm_sd = mu, sd

        print("特征矩阵形状:", self.features.shape)  # (N_day, 10)
        
    def _create_sequences(self):
        """创建滑动窗口序列；推理模式允许目标日无标签（label=-1）"""
        sequences_feature_dataset = []
        sequences_label_dataset = []
        target_feature_dataset = []
        labels_dataset = []
        seq_date = []
        
        # 用“全日期轴”对齐标签（推理模式下可能含 NaN）
        label_values_full = self.label_data.reindex(self.date_index).to_numpy()
        
        for i in range(len(self.date_index)):
            if i < self.sequence_length:
                continue

            # 历史 7 天
            seq_features = self.features[i-self.sequence_length:i]         # (7, 10)
            seq_labels   = label_values_full[i-self.sequence_length:i]     # (7,)
            seq_labels   = np.nan_to_num(seq_labels, nan=0).astype(int)    # 推理时可能有 NaN → 置 0

            # 目标日
            target_feature = self.features[i, self.mask_index]             # (6,)
            target_label   = label_values_full[i]                          # 可能 NaN

            if np.isnan(target_label):
                if not self.inference_allow_unlabeled:
                    continue
                target_label = -1   # 推理占位

            seq_date.append(self.date_index[i])
            sequences_feature_dataset.append(seq_features)
            target_feature_dataset.append(target_feature)
            labels_dataset.append(int(target_label))
            sequences_label_dataset.append(seq_labels)
            
        self.sequences_feature_dataset = np.array(sequences_feature_dataset)
        self.sequences_label_dataset   = np.array(sequences_label_dataset)
        self.target_feature_dataset    = np.array(target_feature_dataset)
        self.labels_dataset            = np.array(labels_dataset)
        self.seq_date                  = np.array(seq_date)
        
        print(f"创建了{len(self.sequences_feature_dataset)}个历史序列样本")
        print(f"创建了{len(self.target_feature_dataset)}个目标特征样本")
        print(f"序列形状: {self.sequences_feature_dataset.shape}")   # (N, 7, 10)
        print(f"目标特征形状: {self.target_feature_dataset.shape}")   # (N, 6)
        # 标签分布（可能包含 -1）
        unique, counts = np.unique(self.labels_dataset, return_counts=True)
        print("标签分布:", dict(zip(unique.tolist(), counts.tolist())))
        
    def _balance_samples(self):
        """平衡正负样本比例，使正负样本比例为1:1（仅训练/验证调用）"""
        positive_indices = np.where(self.labels_dataset == 1)[0]
        negative_indices = np.where(self.labels_dataset == 0)[0]
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        
        print(f"原始样本分布 - 正样本: {n_positive}, 负样本: {n_negative}")
        
        if n_positive == 0 or n_negative == 0:
            print("警告：只有一种类型的样本，无法进行平衡")
            return
            
        target_count = min(n_positive, n_negative) * 2
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        balanced_positive_indices = positive_indices[:target_count // 2]
        balanced_negative_indices = negative_indices[:target_count // 2]
        balanced_indices = np.concatenate([balanced_positive_indices, balanced_negative_indices])
        np.random.shuffle(balanced_indices)
        
        self.sequences_feature_dataset = self.sequences_feature_dataset[balanced_indices]
        self.sequences_label_dataset   = self.sequences_label_dataset[balanced_indices]
        self.target_feature_dataset    = self.target_feature_dataset[balanced_indices]
        self.labels_dataset            = self.labels_dataset[balanced_indices]
        self.seq_date                  = self.seq_date[balanced_indices]
        
        print(f"样本平衡后 - 总样本数: {len(self.labels_dataset)}")
        print(f"平衡后标签分布: {np.bincount(self.labels_dataset.astype(int))}")
        
    def _enhance_samples(self, noise_std=0.01, time_shift_range=1):
        """
        使用数据增强技术增强正样本（仅训练/验证调用）
        """
        positive_indices = np.where(self.labels_dataset == 1)[0]
        if len(positive_indices) == 0:
            print("没有正样本可以增强")
            return
            
        enhanced_sequences = []
        enhanced_target_features = []
        enhanced_labels = []
        enhanced_seq_labels = []
        enhanced_dates = []
        
        for idx in positive_indices:
            seq_feature = self.sequences_feature_dataset[idx].copy()
            target_feature = self.target_feature_dataset[idx].copy()
            seq_label = self.sequences_label_dataset[idx].copy()
            label = int(self.labels_dataset[idx])
            date = self.seq_date[idx]
            
            # 1. 高斯噪声
            noise = np.random.normal(0, noise_std, seq_feature.shape)
            enhanced_seq_feature = seq_feature + noise
            target_noise = np.random.normal(0, noise_std, target_feature.shape)
            enhanced_target_feature = target_feature + target_noise
            
            enhanced_sequences.append(enhanced_seq_feature)
            enhanced_target_features.append(enhanced_target_feature)
            enhanced_labels.append(label)
            enhanced_seq_labels.append(seq_label)
            enhanced_dates.append(date)
            
            # 2. 时间偏移
            if np.random.random() < 0.5:
                shift = np.random.randint(-time_shift_range, time_shift_range + 1)
                if shift != 0:
                    shifted_seq_feature = np.roll(seq_feature, shift, axis=1)
                    shifted_target_feature = np.roll(target_feature, shift, axis=0)
                    enhanced_sequences.append(shifted_seq_feature)
                    enhanced_target_features.append(shifted_target_feature)
                    enhanced_labels.append(label)
                    enhanced_seq_labels.append(seq_label)
                    enhanced_dates.append(date)
        
        if enhanced_sequences:
            self.sequences_feature_dataset = np.concatenate([self.sequences_feature_dataset, np.array(enhanced_sequences)], axis=0)
            self.target_feature_dataset    = np.concatenate([self.target_feature_dataset, np.array(enhanced_target_features)], axis=0)
            self.labels_dataset            = np.concatenate([self.labels_dataset, np.array(enhanced_labels)], axis=0)
            self.sequences_label_dataset   = np.concatenate([self.sequences_label_dataset, np.array(enhanced_seq_labels)], axis=0)
            self.seq_date                  = np.concatenate([self.seq_date, np.array(enhanced_dates)], axis=0)
            
            print(f"数据增强后 - 总样本数: {len(self.labels_dataset)}")
            print(f"增强后标签分布: {np.bincount(self.labels_dataset.astype(int), minlength=2)}")
        
    def __len__(self):
        return len(self.labels_dataset)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences_feature_dataset[idx]),
            torch.FloatTensor(self.target_feature_dataset[idx]),
            torch.LongTensor(self.sequences_label_dataset[idx]),
            torch.LongTensor([int(self.labels_dataset[idx])]),  # 可能为 -1（推理样本）
        )


# 可选：本地快速验证
if __name__ == "__main__":
    dataset = HikingDataset(
        feature_data_path="data/processed/shanxi_new.parquet",
        label_data_path="data/processed/hiking_01_dataset.csv",
        sequence_length=7,
        start_time="2025-01-01",
        end_time="2025-01-31",
        balance_samples=True,
        use_data_augmentation=False,
        inference_allow_unlabeled=False  # 训练/验证
    )
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dl:
        seqs, tgt, seq_lbls, lbls = batch
        print("seqs:", seqs.shape, "tgt:", tgt.shape, "seq_lbls:", seq_lbls.shape, "lbls:", lbls.shape)
        break
