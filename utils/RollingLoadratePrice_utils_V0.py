import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter("ignore", category=SettingWithCopyWarning)

import sys
#sys.path.append("../")
from ElecPriceCurve_utils import *
from date_utils import *


from typing import Dict, Any
import pandas as pd
import numpy as np

class RollingLoadratePriceForecaster:
    """
    用前 d 天的负荷率-价格数据拟合'负荷率→价格'静态关系，
    再用当天负荷率预测当天价格。保持与原 curves_record 相同结构。
    """
    def __init__(
        self,
        market_data: pd.DataFrame,
        bin_interval: int = 3,
        max_load_rate: int = 100,
        new_interval: int = 1,
        col_map: Dict[str, str] = None,
        require_min_bins: int = 3  # 至少需要多少个有效分箱点才能插值
    ):
        self.market_data = market_data.sort_index()  # 仅排序，不改内容
        self.bin_interval = bin_interval
        self.max_load_rate = max_load_rate
        self.new_interval = new_interval
        self.require_min_bins = require_min_bins
        
        # 列名映射，默认与你之前脚本一致
        default_map = {
            "loadrate_da": "日前负荷率",
            "loadrate_rt": "实时负荷率",
            "price_da": "日前价格",    # 你之前已设为 market_data["price"]
            "price_rt": "实时价格",
            "bidspace_da": "日前竞价空间",
            "bidspace_rt": "实时竞价空间",
        }
        self.col = default_map if col_map is None else col_map

    def _concat_past_days(self, anchor_date: pd.Timestamp, days: int, col: str) -> np.ndarray:
        """拼接 anchor_date 之前 d 天的某一列的当天全时段数据"""
        arrs = []
        for k in range(1, days + 1):
            day_str = (anchor_date - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            try:
                s = self.market_data.loc[day_str, col]
            except KeyError:
                continue  # 没这天就跳过
            # s 可能是 Series（一天多时段），也可能是标量（极少见），统一成 np.array
            vals = s.values if hasattr(s, "values") else np.array([s])
            # 过滤 NaN
            vals = vals[~pd.isna(vals)]
            if vals.size:
                arrs.append(vals)
        if arrs:
            return np.concatenate(arrs)
        return np.array([])

    def _fit_curve_from_history(self, x_hist: np.ndarray, y_hist: np.ndarray):
        """用历史 x,y 拟合分箱曲线并插值；若有效点太少则返回 None"""
        if x_hist.size == 0 or y_hist.size == 0:
            return None, None
        price_mean_by_bin = bin_curve_model(
            load_rate=pd.Series(x_hist),
            price=pd.Series(y_hist),
            bin_interval=self.bin_interval,
            max_load_rate=self.max_load_rate
        )
        # 有效分箱点太少，无法用你现有的二次插值稳妥插值
        if price_mean_by_bin.dropna().shape[0] < self.require_min_bins:
            return price_mean_by_bin, None
        curve_interp = interpolate_bin_curve(price_mean_by_bin, new_interval=self.new_interval)
        return price_mean_by_bin, curve_interp

    def fit_predict(
        self,
        start_date: str,
        end_date: str,
        d: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        返回与原来相同结构的 curves_record（key 为 'YYYY-MM-DD'）。
        每天的预测价格由“前 d 天”拟合出来的曲线得出。
        """
        curves_record: Dict[str, Dict[str, Any]] = {}
        for date in pd.date_range(start=start_date, end=end_date, freq="D"):
            date_str = date.strftime("%Y-%m-%d")

            # 当天数据（用于回代预测 & 画对比）
            try:
                lr_da_today = self.market_data.loc[date_str, self.col["loadrate_da"]].values
                lr_rt_today = self.market_data.loc[date_str, self.col["loadrate_rt"]].values
                price_da_today = self.market_data.loc[date_str, self.col["price_da"]].values
                price_rt_today = self.market_data.loc[date_str, self.col["price_rt"]].values
            except KeyError:
                # 当天没数据就跳过
                continue

            # 历史窗口（不含当天）
            lr_da_hist = self._concat_past_days(date, d, self.col["loadrate_da"])
            pr_da_hist = self._concat_past_days(date, d, self.col["price_da"])
            lr_rt_hist = self._concat_past_days(date, d, self.col["loadrate_rt"])
            pr_rt_hist = self._concat_past_days(date, d, self.col["price_rt"])

            # 拟合历史曲线并插值
            curve_da_bin, curve_da_interp = self._fit_curve_from_history(lr_da_hist, pr_da_hist)
            curve_rt_bin, curve_rt_interp = self._fit_curve_from_history(lr_rt_hist, pr_rt_hist)

            # 用“历史曲线（插值后若可用，否则退回分箱曲线）”回代今天负荷率，预测今天价格
            if curve_da_interp is not None:
                price_pred_da = find_y_vectorized(lr_da_today, curve_da_interp)
            elif curve_da_bin is not None:
                price_pred_da = find_y_vectorized(lr_da_today, curve_da_bin)
            else:
                price_pred_da = pd.Series([np.nan] * len(lr_da_today))

            if curve_rt_interp is not None:
                price_pred_rt = find_y_vectorized(lr_rt_today, curve_rt_interp)
            elif curve_rt_bin is not None:
                price_pred_rt = find_y_vectorized(lr_rt_today, curve_rt_bin)
            else:
                price_pred_rt = pd.Series([np.nan] * len(lr_rt_today))

            # 组织成和你原代码一致的字典
            curves = {
                # 这些字段方便你后续诊断，绘图只用 price_pred_* / price_* 即可
                "bidspace_da": self.market_data.loc[date_str, self.col["bidspace_da"]].values \
                               if self.col["bidspace_da"] in self.market_data.columns else None,
                "bidspace_rt": self.market_data.loc[date_str, self.col["bidspace_rt"]].values \
                               if self.col["bidspace_rt"] in self.market_data.columns else None,
                "loadrate_da": lr_da_today,
                "loadrate_rt": lr_rt_today,
                "price_da": price_da_today,
                "price_rt": price_rt_today,

                # 历史拟合得到的静态曲线
                "loadrate_price_bin_curve_da": curve_da_bin,
                "loadrate_price_bin_curve_rt": curve_rt_bin,
                "loadrate_price_bin_curve_interpolate_da": curve_da_interp,
                "loadrate_price_bin_curve_interpolate_rt": curve_rt_interp,

                # ——最重要：用“历史曲线”对“今天”负荷率的预测——
                "price_pred_da": price_pred_da,
                "price_pred_rt": price_pred_rt,
            }

            curves_record[date_str] = curves

        return curves_record




# ====== 便捷函数：一行产出 curves_record（可直接接你原来的绘图代码） ======
def build_curves_record_with_history(
    market_data: pd.DataFrame,
    date_begin: str,
    date_end: str,
    d: int,
    bin_interval: int = 3,
    max_load_rate: int = 100,
    new_interval: int = 1,
) -> Dict[str, Dict[str, Any]]:
    forecaster = RollingLoadratePriceForecaster(
        market_data=market_data,
        bin_interval=bin_interval,
        max_load_rate=max_load_rate,
        new_interval=new_interval
    )
    return forecaster.fit_predict(date_begin, date_end, d)



# ====== 便捷函数：初步处理市场数据（更改列名、计算负荷率等），与之前的当日预测代码一致 ======
def process_market_data(market_data, date_begin, date_end):
    '''
    初步处理市场数据，更改数据名字以及更换时间。与之前的当日预测的代码一致
    '''
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

    market_data["日前竞价空间"] = pred_space
    market_data["实时竞价空间"] = real_space

    market_data["实时负荷率"] = real_loadrate
    market_data["日前负荷率"] = pred_loadrate

    market_data["日前价格"] = market_data["price"]

    return market_data



# ====== 便捷函数：绘图（与之前的当日预测代码一致） ======
def plot_compare_predict_vs_true(curves_record: Dict[str, Dict[str, Any]]):
    '''
    绘制预测价格与真实价格对比图，与之前的当日预测的代码一致
    '''
    import math

    # 每行4个子图
    n_cols = 6
    n_rows = math.ceil(len(curves_record) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=False)

    # 保证axes是二维数组
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (date_str, curves) in enumerate(curves_record.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        # 绘制预测价格和真实价格
        ax.plot(curves["price_pred_da"].values, label="预测价格-日前", linestyle='--', color='blue')
        ax.plot(curves["price_da"], label="真实价格-日前", linestyle='-', color='blue', alpha=0.5)
        ax.plot(curves["price_pred_rt"].values, label="预测价格-实时", linestyle='--', color='red')
        ax.plot(curves["price_rt"], label="真实价格-实时", linestyle='-', color='red', alpha=0.5)
        ax.set_title(f"{date_str} 预测与真实价格对比", fontsize=9)
        ax.set_ylabel("价格")
        ax.legend()
        ax.grid(True)

    # 隐藏多余的子图
    total_plots = n_rows * n_cols
    for idx in range(len(curves_record), total_plots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.xlabel("时刻")
    plt.show()



def plot_loadrate_price_curves(curves_record: Dict[str, Dict[str, Any]]):
    '''
    绘制量价曲线，与之前的当日预测的代码基本一致。增加了可能存在的“历史不足，无法拟合”的提示。
    '''

    import matplotlib.pyplot as plt

    import math

    # 每行6个子图
    n_cols = 6
    n_rows = math.ceil(len(curves_record) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)

    # 保证axes是二维数组
    axes = np.array(axes).reshape(n_rows, n_cols)


    for idx, (date_str, curves) in enumerate(curves_record.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 取出插值后的量价曲线（可能为 None，所以加兜底）
        curve_da = curves.get("loadrate_price_bin_curve_interpolate_da")
        curve_rt = curves.get("loadrate_price_bin_curve_interpolate_rt")

        if isinstance(curve_da, pd.Series) and isinstance(curve_da.index, pd.IntervalIndex):
            x_da = np.fromiter((iv.left for iv in curve_da.index), dtype=float)
            ax.plot(x_da, curve_da.values, label="日前量价曲线", color="blue")
        else:
            ax.text(0.5, 0.55, "历史不足，无法拟合DA", transform=ax.transAxes, ha="center", va="center")

        if isinstance(curve_rt, pd.Series) and isinstance(curve_rt.index, pd.IntervalIndex):
            x_rt = np.fromiter((iv.left for iv in curve_rt.index), dtype=float)
            ax.plot(x_rt, curve_rt.values, label="实时量价曲线", color="red")
        else:
            ax.text(0.5, 0.40, "历史不足，无法拟合RT", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(f"{date_str}")
        ax.set_xlabel("负荷率(%)")
        ax.set_ylabel("价格")
        ax.set_ylim(0, 1400)
        ax.legend()
        ax.grid(True)


    # 隐藏多余的子图
    total_plots = n_rows * n_cols
    for idx in range(len(curves_record), total_plots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
