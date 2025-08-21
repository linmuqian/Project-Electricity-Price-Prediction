import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


date_begin = "2023-01-01"
date_end = date.today().strftime("%Y-%m-%d")

#bid_space_dataset = pd.read_parquet("../data/processed/da_bidspace/bidspace_dataset.parquet",engine='fastparquet')
#bid_space_dataset.index = pd.to_datetime(bid_space_dataset.index)
#bid_space_dataset = bid_space_dataset[date_begin:date_end]

market_data = pd.read_parquet("data/processed/shanxi_new.parquet",engine='fastparquet')
market_data.index = pd.to_datetime(market_data.index)

market_data = market_data[date_begin:date_end]
#market_data["日前推测竞价空间"] = bid_space_dataset["推测竞价空间"]
#market_data["日前推测负荷率"] = market_data["日前推测竞价空间"]/market_data["日前在线机组容量(MW)"]*100
print(market_data.columns)

pred_space = (
    market_data["省调负荷-日前(MW)"]
    - market_data["新能源负荷-日前(MW)"]
    + market_data["联络线计划-日前(MW)"]
    - market_data["非市场化机组出力-日前(MW)"]
)
real_space = (
    market_data["省调负荷-日内(MW)"]
    - market_data["新能源负荷-日内(MW)"]
    + market_data["联络线计划-日内(MW)"]
    - market_data["非市场化机组出力-日内(MW)"]
)
pred_loadrate = pred_space/market_data["日前在线机组容量(MW)"]*100
real_loadrate = real_space/market_data["实时在线机组容量(MW)"]*100

pred_capacity = market_data["日前在线机组容量(MW)"]
real_capacity = market_data["实时在线机组容量(MW)"]

market_data["日前竞价空间"] = pred_space
market_data["实时竞价空间"] = real_space

market_data["实时负荷率"] = real_loadrate
market_data["日前负荷率"] = pred_loadrate

market_data["日前价格"] = market_data["price"]


import sys
#sys.path.append("../")
from utils.ElecPriceCurve_utils import *
from utils.date_utils import *

curves_record = {}

for date in pd.date_range(start=date_begin, end=date_end, freq="D"):
    date_str = date.strftime("%Y-%m-%d")
    curves = {
        "bidspace_da": market_data.loc[date_str, '日前竞价空间'].values,
        "bidspace_rt": market_data.loc[date_str, '实时竞价空间'].values,
        "loadrate_da": market_data.loc[date_str, '日前负荷率'].values,
        "loadrate_rt": market_data.loc[date_str, '实时负荷率'].values,
        "price_da": market_data.loc[date_str, '日前价格'].values,
        "price_rt": market_data.loc[date_str, '实时价格'].values,
        "capacity_da": market_data.loc[date_str, '日前在线机组容量(MW)'].values,
        "capacity_rt": market_data.loc[date_str, '实时在线机组容量(MW)'].values,
    }
    # 针对不同bin_interval分别拟合量价曲线并记录
    for bin_interval in range(3, 4):  # 1~5
        # 日前
        bin_curve_da = bin_curve_model(
            curves["loadrate_da"],
            curves["price_da"],
            bin_interval=bin_interval,
            max_load_rate=100
        )
        bin_curve_da_interp = interpolate_bin_curve(
            bin_curve_da,
            new_interval=1
        )
        price_pred_da = find_y_vectorized(curves["loadrate_da"], bin_curve_da_interp)
        curves[f"loadrate_price_bin_curve_da_{bin_interval}"] = bin_curve_da
        curves[f"loadrate_price_bin_curve_interpolate_da_{bin_interval}"] = bin_curve_da_interp
        curves[f"price_pred_da_{bin_interval}"] = price_pred_da

        # 日内
        bin_curve_rt = bin_curve_model(
            curves["loadrate_rt"],
            curves["price_rt"],
            bin_interval=bin_interval,
            max_load_rate=100
        )
        bin_curve_rt_interp = interpolate_bin_curve(
            bin_curve_rt,
            new_interval=1
        )
        price_pred_rt = find_y_vectorized(curves["loadrate_rt"], bin_curve_rt_interp)
        curves[f"loadrate_price_bin_curve_rt_{bin_interval}"] = bin_curve_rt
        curves[f"loadrate_price_bin_curve_interpolate_rt_{bin_interval}"] = bin_curve_rt_interp
        curves[f"price_pred_rt_{bin_interval}"] = price_pred_rt

    curves_record[date_str] = curves


## 生成数据集
hiking_df = {}
date_list = []
da_hiking_list = []
rt_hiking_list = []

for date in curves_record.keys():
    date_list.append(date)
    da_hiking_list.append(curves_record[date]["loadrate_price_bin_curve_interpolate_da_3"].values[-1])
    rt_hiking_list.append(curves_record[date]["loadrate_price_bin_curve_interpolate_rt_3"].values[-1])

hiking_df["date"] = date_list
hiking_df["da_hiking"] = da_hiking_list
hiking_df["rt_hiking"] = rt_hiking_list
hiking_df = pd.DataFrame(hiking_df)
hiking_df.to_parquet("data/processed/hiking_dataset.parquet",engine='fastparquet')
hiking_df.to_csv("data/processed/hiking_dataset.csv",index=False)


## 生成数据集
hiking_df = {}
date_list = []
da_hiking_list = []
rt_hiking_list = []

for date in curves_record.keys():
    date_list.append(date)
    da_hiking_list.append(curves_record[date]["loadrate_price_bin_curve_interpolate_da_3"].values[-1]>=500)
    rt_hiking_list.append(curves_record[date]["loadrate_price_bin_curve_interpolate_rt_3"].values[-1]>=500)

hiking_df["date"] = date_list
hiking_df["da_hiking"] = da_hiking_list
hiking_df["rt_hiking"] = rt_hiking_list
hiking_df = pd.DataFrame(hiking_df)
hiking_df.to_parquet("data/processed/hiking_01_dataset.parquet",engine='fastparquet')
hiking_df.to_csv("data/processed/hiking_01_dataset.csv",index=False)

print("数据集生成完毕！")