import os
import sys
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
# 添加项目根目录到路径（加载父目录的包）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_single_GRU import main as train_single_GRU


def main():
    train_begin = "2024-10-01"
    train_end = "2025-01-31"
    valid_begin = "2025-02-01"
    valid_end = "2025-02-14"
    test_begin = "2025-02-15"
    test_end = "2025-02-21"
    
    date_end = "2025-08-08"
    rolling_interval = 7
    record_predict_df = []
    while pd.to_datetime(test_begin) < pd.to_datetime(date_end):
        predict_df = train_single_GRU(train_begin, train_end, valid_begin, valid_end, test_begin, test_end)
        record_predict_df.append(predict_df)
        train_begin = (pd.to_datetime(train_begin) + pd.Timedelta(days=rolling_interval)).strftime("%Y-%m-%d")
        train_end = (pd.to_datetime(train_end) + pd.Timedelta(days=rolling_interval)).strftime("%Y-%m-%d")
        valid_begin = (pd.to_datetime(valid_begin) + pd.Timedelta(days=rolling_interval)).strftime("%Y-%m-%d")
        valid_end = (pd.to_datetime(valid_end) + pd.Timedelta(days=rolling_interval)).strftime("%Y-%m-%d")
        test_begin = (pd.to_datetime(test_begin) + pd.Timedelta(days=rolling_interval)).strftime("%Y-%m-%d")
        test_end = pd.to_datetime(test_end) + pd.Timedelta(days=rolling_interval)
        test_end = min(test_end, pd.to_datetime(date_end))
        test_end = test_end.strftime("%Y-%m-%d")
    record_predict_df = pd.concat(record_predict_df)
    record_predict_df.to_csv("save/curve_classify/multi_GRU_predict_results.csv",index=False)

if __name__ == "__main__":
    main()
