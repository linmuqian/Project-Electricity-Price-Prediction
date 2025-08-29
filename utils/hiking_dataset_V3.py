# hiking_dataset_V3.py  —— 针对V2版本修改了数据标准化的方式
'''
针对针对V2版本修改了数据标准化的方式
即【price】相关列（0,5）分列标准化
【日前负荷率】两列（1,6）直接置 0
【日前风电】【省调负荷】【竞价空间】（2,3,4,7,8,9）做全局标准化（一个共同的 μ、σ）

V2:【省调负荷-日前】和【竞价空间-日前】都给替换成当日-前d日均值
V2-1:【省调负荷-日前】和【竞价空间-日前】保留，再加入(当日- Hiking=0前d日均值)和(当日- Hiking=1前d日均值)
V2-2:【省调负荷-日前】和【竞价空间-日前】不保留，加入(当日- Hiking=0前d日均值)和(当日- Hiking=1前d日均值)
V2-3:【日前风电】【省调负荷-日前】和【竞价空间-日前】都给替换成当日-前d日均值
V2-4:【省调负荷-日前】和【竞价空间-日前】不保留，替换成 (当日- Hiking=0前d日均值)
'''

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class HikingDataset(Dataset):
    """抬价预测数据集（V2：调整特征构造；支持推理模式） 
       - inference_allow_unlabeled=False：训练/验证（只保留有标签的目标日）
       - inference_allow_unlabeled=True ：推理（允许目标日无标签，label 用 -1 占位）

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

    def __init__(self, feature_data_path: str,
                 label_data_path: str,
                 sequence_length: int = 7,
                 feature_dim: int = 24,
                 start_time: str = None,
                 end_time: str = None,
                 balance_samples: bool = False,
                 use_data_augmentation: bool = False,
                 inference_allow_unlabeled: bool = False,   # [MOD] 推理开关：目标日可无标签
                 norm_mu: np.ndarray = None,                # [MOD] 传入训练集拟合的均值(逐列)，防止泄露
                 norm_sd: np.ndarray = None):               # [MOD] 传入训练集拟合的方差(逐列)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.balance_samples = balance_samples
        self.use_data_augmentation = use_data_augmentation
        self.inference_allow_unlabeled = inference_allow_unlabeled
        self.norm_mu = norm_mu
        self.norm_sd = norm_sd

        # 读取特征数据（15min 频率）
        market_data = pd.read_parquet(feature_data_path, engine='fastparquet')
        market_data.index = pd.to_datetime(market_data.index)
        self.market_data = market_data

        # 读取标签数据（天级）
        label_data = pd.read_csv(label_data_path)
        label_data['date'] = pd.to_datetime(label_data['date'])
        label_data.set_index('date', inplace=True)
        # self.label_data = label_data["da_hiking"].astype(int)  # 原：整型不支持 NaN
        self.label_data = pd.to_numeric(label_data["da_hiking"], errors="coerce")  # [MOD] 允许 NaN，便于推理

        # 日期轴选择
        if not self.inference_allow_unlabeled:
            # 训练/验证：裁到标签范围，按标签日期建轴
            self.market_data = self.market_data.loc[
                self.label_data.index[0].strftime("%Y-%m-%d"):
                self.label_data.index[-1].strftime("%Y-%m-%d")
            ]
            self.date_index = self.label_data.index
        else:
            # 推理：裁到标签范围，按特征日期建轴（允许目标日无标签）
            self.market_data = self.market_data.loc[
                self.label_data.index[0].strftime("%Y-%m-%d"):
                self.market_data.index[-1].strftime("%Y-%m-%d")
            ]
            self.date_index = pd.DatetimeIndex(pd.to_datetime(self.market_data.index).normalize().unique()).sort_values()

        print(f"数据加载完成，共{len(self.label_data)}条标签记录")
        print(f"特征列数: {len(market_data.columns)}")
        print(f"标签时间范围: {self.label_data.index.min()} 到 {self.label_data.index.max()}")

        # 1) 创建特征（V2）
        self._create_features()
        # 2) 创建序列数据
        self._create_sequences()

        # 3) 可选时间窗过滤
        if start_time and end_time:
            mask = (self.seq_date >= pd.to_datetime(start_time)) & (self.seq_date <= pd.to_datetime(end_time))
            self.seq_date = self.seq_date[mask]
            self.sequences_feature_dataset = self.sequences_feature_dataset[mask]
            self.sequences_label_dataset = self.sequences_label_dataset[mask]
            self.target_feature_dataset = self.target_feature_dataset[mask]
            self.labels_dataset = self.labels_dataset[mask]

        # 4) 仅训练/验证阶段做平衡与增强（推理不做）
        if (not self.inference_allow_unlabeled) and self.balance_samples:
            self._balance_samples()
        if (not self.inference_allow_unlabeled) and self.use_data_augmentation:
            self._enhance_samples()

    def _create_features(self):
        """从真实数据创建特征矩阵（V2：对列3/4做“当前 - 前D天均值”差分；不依赖标签）"""
        cols = [
            "price",                 # 0
            "日前负荷率(%)",          # 1
            "日前风电(MW)",           # 2
            "省调负荷-日前(MW)",       # 3 ← 做差分(连续窗口)
            "竞价空间-日前(MW)"        # 4 ← 做差分(连续窗口)
        ]
        self.elec_feature = self.market_data[cols].copy()
        n_feat = len(cols)

        # 每日 96 条校验
        if len(self.elec_feature) % 96 != 0:
            raise ValueError("原始特征行数必须是96的整数倍（每日96条15min数据）")

        D = 5            # 前 d 天
        STEPS = 96
        NIGHT_BEG = 64   # 16:00 → 索引 64（窗口 [64:96)）

        # 形状：(N_day, 96, n_feat)
        X = self.elec_feature.values.reshape(-1, STEPS, n_feat)
        night = X[:, NIGHT_BEG:, :]  # 夜间窗口

        # [MOD] NaN 安全统计（避免全 NaN 传播）
        day_max  = np.nanmax(night, axis=1)   # (N_day, n_feat)
        day_mean = np.nanmean(night, axis=1)

        # [MOD] 若整列存在 NaN，用列内前后值填充，兜底为 0
        for arr in (day_max, day_mean):
            for j in range(arr.shape[1]):
                s = pd.Series(arr[:, j])
                if s.isna().any():
                    arr[:, j] = s.ffill().bfill().fillna(0.0).to_numpy()

        # 辅助函数：计算 “当前 - 前D天均值（不含当天）”，连续时间窗
        def _sub_prev_d_mean(col: np.ndarray, d: int = D) -> np.ndarray:
            s = pd.Series(col)
            prev_mean = s.shift(1).rolling(window=d, min_periods=1).mean()
            prev_mean = prev_mean.fillna(s)  # 首日/不足时用自身，差为0
            return (s - prev_mean).to_numpy()

        # 仅对列3/4（省调/竞价）的 max/mean 做差分
        for j in (3, 4):
            day_max[:,  j] = _sub_prev_d_mean(day_max[:,  j], D)
            day_mean[:, j] = _sub_prev_d_mean(day_mean[:, j], D)

        # 拼成 10 维：[max5 || mean5]
        self.elec_feature = np.concatenate([day_max, day_mean], axis=1)

        # 目标日不可用 price/负荷率 的 max/mean（索引 [0,1,5,6]），目标特征=6维
        self.mask_index = np.ones(self.elec_feature.shape[1], dtype=bool)
        self.mask_index[[0, 1, 5, 6]] = False

        # [MOD] 标准化：nanmean/nanstd；允许外部传入(防泄露)
        f = self.elec_feature.astype(float)                         # (N_day, 10)
        '''
        mu = self.norm_mu if self.norm_mu is not None else f.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)
        sd = self.norm_sd if self.norm_sd is not None else f.std(axis=0, keepdims=True).std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        f = (f - mu) / sd
        f[~np.isfinite(f)] = 0.0                                   # 清理 NaN/Inf

        # 如果不使用【日前负荷率】的两个特征的话
        # f[:, [1,6,3,8,0,5]] = 0.0

        self.features = f
        self.norm_mu, self.norm_sd = mu, sd                        # 保存以便复用
        '''
                # ===== [CHANGE] 标准化策略重写 =====
        # 需求：
        #  - price 相关（列 0,5）分别按列标准化（各自 z-score）
        #  - 【日前负荷率】两列（1,6）直接置 0
        #  - 【日前风电】【省调负荷】【竞价空间】这 6 列（2,3,4,7,8,9）一起做全局标准化（一个共同的 μ、σ）

        price_cols = [0, 5]
        rate_cols  = [1, 6]                  # 置 0
        group_cols = [2, 3, 4, 7, 8, 9]      # 6 列共用一个 μ、σ

        # --- price: 分列标准化 ---
        # 兼容：若传入 self.norm_mu / self.norm_sd 为 dict，可复用训练集统计量以防泄露
        p_mu = None; p_sd = None
        if isinstance(self.norm_mu, dict) and isinstance(self.norm_sd, dict):
            p_mu = self.norm_mu.get('price', None)  # 期望形状 (2,)
            p_sd = self.norm_sd.get('price', None)

        if p_mu is None or p_sd is None:
            p_mu = np.nanmean(f[:, price_cols], axis=0)     # (2,)
            p_sd = np.nanstd (f[:, price_cols], axis=0)     # (2,)
        p_sd = np.where(p_sd == 0, 1.0, p_sd)
        f[:, price_cols] = (f[:, price_cols] - p_mu) / p_sd

        # --- 日前负荷率: 直接置 0（置 0 放在 group 统计之前，避免干扰 group 的 μ、σ）---
        f[:, rate_cols] = 0.0

        # --- 风电/省调负荷/竞价空间: 6 列合并全局标准化（一个共同的 μ、σ）---
        g_mu = None; g_sd = None
        if isinstance(self.norm_mu, dict) and isinstance(self.norm_sd, dict):
            g_mu = self.norm_mu.get('group', None)          # 标量
            g_sd = self.norm_sd.get('group', None)

        if g_mu is None or g_sd is None:
            g_mu = np.nanmean(f[:, group_cols])             # 标量
            g_sd = np.nanstd (f[:, group_cols])             # 标量
        if not np.isfinite(g_sd) or g_sd == 0:
            g_sd = 1.0
        f[:, group_cols] = (f[:, group_cols] - g_mu) / g_sd

        # 清理数值
        f[~np.isfinite(f)] = 0.0

        self.features = f
        # 将拟合到的统计量保存/回传，便于验证/推理阶段复用（防止泄露）
        self.norm_mu = {'price': p_mu, 'group': g_mu}
        self.norm_sd = {'price': p_sd, 'group': g_sd}

        print("特征矩阵形状:", self.features.shape)  # (N_day, 10)

    def _create_sequences(self):
        """创建滑动窗口序列；推理模式允许目标日无标签（label=-1）"""
        seq_X, seq_hist_y, tgt_X, tgt_y, seq_date = [], [], [], [], []

        # 用全日期轴对齐标签（推理模式下可能出现 NaN）
        label_values_full = self.label_data.reindex(self.date_index).to_numpy()

        for i in range(len(self.date_index)):
            if i < self.sequence_length:
                continue

            # 历史 7 天
            hist_feat = self.features[i - self.sequence_length: i]              # (7, 10)
            hist_lbl  = label_values_full[i - self.sequence_length: i]          # (7,)
            hist_lbl  = np.nan_to_num(hist_lbl, nan=0).astype(int)              # 缺失置 0

            # 目标日
            target_feat = self.features[i, self.mask_index]                     # (6,)
            target_lbl  = label_values_full[i]                                  # 可能 NaN

            if np.isnan(target_lbl):
                if not self.inference_allow_unlabeled:
                    continue
                target_lbl = -1  # 推理占位

            seq_X.append(hist_feat)
            seq_hist_y.append(hist_lbl)
            tgt_X.append(target_feat)
            tgt_y.append(int(target_lbl))
            seq_date.append(self.date_index[i])

        self.sequences_feature_dataset = np.array(seq_X)
        self.sequences_label_dataset   = np.array(seq_hist_y)
        self.target_feature_dataset    = np.array(tgt_X)
        self.labels_dataset            = np.array(tgt_y)
        self.seq_date                  = np.array(seq_date)

        print(f"创建了{len(self.sequences_feature_dataset)}个历史序列样本")
        print(f"创建了{len(self.target_feature_dataset)}个目标特征样本")
        print(f"序列形状: {self.sequences_feature_dataset.shape}")   # (N, 7, 10)
        print(f"目标特征形状: {self.target_feature_dataset.shape}")   # (N, 6)
        uniq, cnt = np.unique(self.labels_dataset, return_counts=True)
        print("标签分布:", dict(zip(uniq.tolist(), cnt.tolist())))

    def _balance_samples(self):
        """平衡正负样本比例，使正负样本比例为1:1（仅训练/验证调用）"""
        pos = np.where(self.labels_dataset == 1)[0]
        neg = np.where(self.labels_dataset == 0)[0]
        n_pos, n_neg = len(pos), len(neg)
        print(f"原始样本分布 - 正样本: {n_pos}, 负样本: {n_neg}")
        if n_pos == 0 or n_neg == 0:
            print("警告：只有一种类型的样本，无法进行平衡")
            return
        n = min(n_pos, n_neg)
        pos = np.random.permutation(pos)[:n]
        neg = np.random.permutation(neg)[:n]
        idx = np.concatenate([pos, neg])
        np.random.shuffle(idx)
        self.sequences_feature_dataset = self.sequences_feature_dataset[idx]
        self.sequences_label_dataset   = self.sequences_label_dataset[idx]
        self.target_feature_dataset    = self.target_feature_dataset[idx]
        self.labels_dataset            = self.labels_dataset[idx]
        self.seq_date                  = self.seq_date[idx]
        print(f"样本平衡后 - 总样本数: {len(self.labels_dataset)}")

    def _enhance_samples(self, noise_std=0.01, time_shift_range=1):
        """仅训练/验证调用的简单增强"""
        pos = np.where(self.labels_dataset == 1)[0]
        if len(pos) == 0:
            print("没有正样本可以增强")
            return
        aug_seq, aug_tgt, aug_lbl, aug_hist_y, aug_date = [], [], [], [], []
        for i in pos:
            s = self.sequences_feature_dataset[i].copy()
            t = self.target_feature_dataset[i].copy()
            y = int(self.labels_dataset[i])
            h = self.sequences_label_dataset[i].copy()
            d = self.seq_date[i]
            # 噪声
            s2 = s + np.random.normal(0, noise_std, s.shape)
            t2 = t + np.random.normal(0, noise_std, t.shape)
            aug_seq.append(s2); aug_tgt.append(t2); aug_lbl.append(y); aug_hist_y.append(h); aug_date.append(d)
            # 时间偏移
            if np.random.rand() < 0.5:
                shift = np.random.randint(-time_shift_range, time_shift_range + 1)
                if shift != 0:
                    aug_seq.append(np.roll(s, shift, axis=1))
                    aug_tgt.append(np.roll(t, shift, axis=0))
                    aug_lbl.append(y); aug_hist_y.append(h); aug_date.append(d)
        if aug_seq:
            self.sequences_feature_dataset = np.concatenate([self.sequences_feature_dataset, np.array(aug_seq)], axis=0)
            self.target_feature_dataset    = np.concatenate([self.target_feature_dataset, np.array(aug_tgt)], axis=0)
            self.labels_dataset            = np.concatenate([self.labels_dataset, np.array(aug_lbl)], axis=0)
            self.sequences_label_dataset   = np.concatenate([self.sequences_label_dataset, np.array(aug_hist_y)], axis=0)
            self.seq_date                  = np.concatenate([self.seq_date, np.array(aug_date)], axis=0)
            print(f"数据增强后 - 总样本数: {len(self.labels_dataset)}")

    def __len__(self): return len(self.labels_dataset)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences_feature_dataset[idx]),  # (7, 10)
            torch.FloatTensor(self.target_feature_dataset[idx]),     # (6,)
            torch.LongTensor(self.sequences_label_dataset[idx]),     # (7,)
            torch.LongTensor([int(self.labels_dataset[idx])]),       # 可能为 -1（推理样本）
        )
