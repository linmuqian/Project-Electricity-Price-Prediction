# -*- coding: utf-8 -*-
"""
RollingLoadratePrice_utils.py

在原先V0“用前 d 天拟合负荷率→价格曲线，再回代当天负荷率预测”的流程上，
新增一个基于“抬价二分类结果”的版本：
  先用预测的 hiking 标签（0/1）判定当天类属，
  再在历史中用“真实标签（更长时域）”筛出前 d 个同类日来拟合曲线。

兼容四种 selection：
  - baseline: 不看任何标签，直接前 d 天（最初的V0版本完全一致）
  - match   : 当天预测=1 → 取历史真实标签=1 的前 d 个日；预测=0 → 取真实=0 的前 d 个
  - pos_only: 无论当天预测什么，历史只取真实=1 的前 d 个
  - neg_only: 同理，只取真实=0 的前 d 个

输出 curves_record 结构与旧版一致，可直接衔接你的两个画图函数：
  plot_compare_predict_vs_true(curves_record)
  plot_loadrate_price_curves(curves_record)
"""

from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm
# myfont = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
# rcParams['font.sans-serif'] = [myfont.get_name()]
# rcParams['axes.unicode_minus'] = False
from matplotlib import rcParams, font_manager as fm

try:
    fams = {f.name for f in fm.fontManager.ttflist}
    # 把 DejaVu 放到 Droid 之前
    order = ["Noto Sans CJK SC", "DejaVu Sans", "Droid Sans Fallback"]  # ← 这里调换顺序
    chosen = [f for f in order if f in fams] or ["DejaVu Sans"]
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = chosen
except Exception:
    pass

rcParams["axes.unicode_minus"] = False



import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter("ignore", category=SettingWithCopyWarning)

# === 你现有的工具函数（保持接口不变） ===
from .ElecPriceCurve_utils import *
from .date_utils import *


# ---------- 通用小工具（CSV -> Series） ----------

def _to01_array(x: pd.Series) -> pd.Series:
    """
    将 True/False / 'True'/'False' / 1/0 / '1'/'0' 安全转成 {0,1}，NaN 保留。
    """
    def _cast(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        s = str(v).strip().lower()
        if s in ("true", "t", "yes", "y"):
            return 1
        if s in ("false", "f", "no", "n"):
            return 0
        try:
            # 兼容 '1.0' 这类
            return int(round(float(s)))
        except Exception:
            return np.nan
    return x.map(_cast)


def load_pred_series_from_csv(
    csv_path: str,
    date_col: str = "Date",
    pred_label_col: str = "Pred_Label",
) -> pd.Series:
    """
    读取图1（预测文件）为日级 Series：
        index: 日期（已 normalize）
        value: 0/1（Pred_Label）
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or pred_label_col not in df.columns:
        raise ValueError(f"预测CSV缺少列：{date_col} 或 {pred_label_col}")
    idx = pd.to_datetime(df[date_col]).dt.normalize()
    val = _to01_array(df[pred_label_col]).astype("float")
    # 去重：同一天多条时取最后一条（也可改成均值/众数）
    s = pd.Series(val.values, index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna().astype(int)
    return s


def load_truth_series_from_csv(
    csv_path: str,
    date_col: str = "date",
    da_col: str = "da_hiking",
    rt_col: str = "rt_hiking",
) -> Tuple[pd.Series, pd.Series]:
    """
    读取图2（真实标签）为两个日级 Series：
        - DA: index=日期, value in {0,1}
        - RT: index=日期, value in {0,1}
    """
    df = pd.read_csv(csv_path)
    for c in (date_col, da_col, rt_col):
        if c not in df.columns:
            raise ValueError(f"真实CSV缺少列：{c}")
    idx = pd.to_datetime(df[date_col]).dt.normalize()
    da = _to01_array(df[da_col]).astype("float")
    rt = _to01_array(df[rt_col]).astype("float")
    s_da = pd.Series(da.values, index=idx).sort_index()
    s_rt = pd.Series(rt.values, index=idx).sort_index()
    # 去重：保留最后一条
    s_da = s_da[~s_da.index.duplicated(keep="last")].dropna().astype(int)
    s_rt = s_rt[~s_rt.index.duplicated(keep="last")].dropna().astype(int)
    return s_da, s_rt


# ---------- 主类：分类驱动的滚动量价曲线 ----------

class RollingLoadratePriceForecaster_classify:
    """
    基于“二分类预测的 hiking 标签（0/1）”挑选历史窗口来拟合‘负荷率→价格’静态关系；
    再用当天负荷率回代预测当天价格。

    selection 选项：
      - "baseline"：不看标签，直接取前 d 天（与你最初“前 d 天”方案完全兼容）。
      - "match"   ：当天预测为 1 → 历史取真实=1 的最近 d 天；预测为 0 → 历史取真实=0 的最近 d 天。
      - "pos_only"：总是取历史真实=1 的最近 d 天。
      - "neg_only"：总是取历史真实=0 的最近 d 天。

    注意：
      - baseline 模式下，不需要预测标签；其余模式需提供“当天的预测标签”。
      - 历史筛选一律用“真实标签”，因为真实 CSV 时间更长、覆盖范围更广。
      - DA/RT 各自独立拟合与预测：DA 用 (loadrate_da, price_da)，RT 用 (loadrate_rt, price_rt)。
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        bin_interval: int = 3,
        max_load_rate: int = 100,
        new_interval: int = 1,
        col_map: Optional[Dict[str, str]] = None,
        require_min_bins: int = 3,  # 至少多少个有效分箱点才能插值
    ):
        # 仅排序，不改内容
        self.market_data = market_data.sort_index()
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

    # ===== 内部工具 =====

    def _concat_days_list(self, days_list: np.ndarray, col: str) -> np.ndarray:
        """拼接若干天（按 'YYYY-MM-DD'）的某一列全天数据为一维历史序列。"""
        arrs = []
        for dstr in days_list:
            try:
                s = self.market_data.loc[dstr, col]
            except KeyError:
                continue  # 没这天就跳过
            vals = s.values if hasattr(s, "values") else np.array([s])
            vals = vals[~pd.isna(vals)]
            if vals.size:
                arrs.append(vals)
        return np.concatenate(arrs) if arrs else np.array([])

    def _fit_curve_from_history(self, x_hist: np.ndarray, y_hist: np.ndarray):
        """用历史 x,y 拟合分箱曲线并插值；若有效点太少则返回 (bin_curve, None)。"""
        if x_hist.size == 0 or y_hist.size == 0:
            return None, None
        price_mean_by_bin = bin_curve_model(
            load_rate=pd.Series(x_hist),
            price=pd.Series(y_hist),
            bin_interval=self.bin_interval,
            max_load_rate=self.max_load_rate,
        )
        # 有效分箱点太少，无法稳妥插值
        if price_mean_by_bin.dropna().shape[0] < self.require_min_bins:
            return price_mean_by_bin, None
        curve_interp = interpolate_bin_curve(price_mean_by_bin, new_interval=self.new_interval)
        return price_mean_by_bin, curve_interp

    def _select_history_dates(
        self,
        all_dates: np.ndarray,                # 日级 DatetimeIndex/np.ndarray
        anchor_date: pd.Timestamp,            # 目标当天
        d: int,
        selection: str,
        # 当天预测标签（来自图1的 CSV）：index=日，val in {0,1}
        pred_label_by_day: Optional[pd.Series] = None,
        # 历史真实标签（来自图2的 CSV）：index=日，val in {0,1}
        true_label_by_day: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        返回历史日期字符串数组（'YYYY-MM-DD'），长度最多 d。
        baseline：直接取 anchor_date 之前的最近 d 天；
        其他模式：按标签过滤后再取最近 d 天（标签来源见函数参数）。
        """
        anchor_day = anchor_date.normalize()
        prev_dates = all_dates[all_dates < anchor_day]  # 不含当天

        if selection == "baseline":
            candidate_dates = prev_dates

        else:
            if true_label_by_day is None:
                raise ValueError("需要提供真实标签CSV来筛选历史日 true_label_by_day。")

            # ——确定“目标标签”
            if selection == "match":
                if pred_label_by_day is None or anchor_day not in pred_label_by_day.index:
                    # 当天没预测标签，无法匹配
                    return np.array([], dtype=str)
                target_label = int(pred_label_by_day.loc[anchor_day])
            elif selection == "pos_only":
                target_label = 1
            elif selection == "neg_only":
                target_label = 0
            else:
                raise ValueError(f"Unknown selection='{selection}' (use 'baseline'|'match'|'pos_only'|'neg_only').")

            # ——用“真实标签”筛选历史（覆盖更长时间）
            prev_truth = true_label_by_day.reindex(prev_dates)
            hist_mask = (prev_truth == target_label).fillna(False).values
            candidate_dates = prev_dates[hist_mask]

        if candidate_dates.size == 0:
            return np.array([], dtype=str)

        selected = candidate_dates[-d:]  # 最近 d 天
        return np.array([pd.Timestamp(x).strftime("%Y-%m-%d") for x in selected])

    # ===== 对外主流程 =====

    def fit_predict(
        self,
        start_date: str,
        end_date: str,
        d: int,
        # 预测标签：用于当天“匹配”目标（图1 CSV → Pred_Label）
        hiking_pred_by_day: Optional[pd.Series] = None,
        # 历史真实标签（图2 CSV）：DA/RT 分别一列（更长时域）
        hiking_truth_da_by_day: Optional[pd.Series] = None,
        hiking_truth_rt_by_day: Optional[pd.Series] = None,
        selection: str = "match",
        horizon: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """
        返回与原来相同结构的 curves_record（key 为 'YYYY-MM-DD'）。
        每天的预测价格由“历史窗口（按 selection 选择）”拟合出来的曲线得出。

        参数：
          - start_date, end_date: 预测区间（闭区间，'YYYY-MM-DD'）
          - d: 取前 d 个历史日
          - hiking_pred_by_day: 每日“预测标签”Series（index=日期；baseline 模式可以为 None）
          - hiking_truth_da_by_day/hiking_truth_rt_by_day: 每日“真实标签”（index=日期）
            * 用于历史日筛选（因为真实CSV覆盖更长时间）
          - selection: 'baseline' | 'match' | 'pos_only' | 'neg_only'
          - horizon: 预测D+horizon天的价格，如horizon=2，则预测D+2天的价格
        """
        
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1")
        
        # 规范索引
        if hiking_pred_by_day is not None:
            hiking_pred_by_day = hiking_pred_by_day.copy()
            hiking_pred_by_day.index = pd.to_datetime(hiking_pred_by_day.index).normalize()
            hiking_pred_by_day = hiking_pred_by_day.sort_index()

        if hiking_truth_da_by_day is not None:
            hiking_truth_da_by_day = hiking_truth_da_by_day.copy()
            hiking_truth_da_by_day.index = pd.to_datetime(hiking_truth_da_by_day.index).normalize()
            hiking_truth_da_by_day = hiking_truth_da_by_day.sort_index()

        if hiking_truth_rt_by_day is not None:
            hiking_truth_rt_by_day = hiking_truth_rt_by_day.copy()
            hiking_truth_rt_by_day.index = pd.to_datetime(hiking_truth_rt_by_day.index).normalize()
            hiking_truth_rt_by_day = hiking_truth_rt_by_day.sort_index()

        # 所有有数据的日期（日级）
        all_dates = pd.to_datetime(self.market_data.index.normalize().unique())
        
        # >>> 新增(Dn)：将预测标签“对齐到基准日”（base_day拿到 target_day 的标签）
        # 如horizon=2时，在base_day=8/25取到的是target_day=8/27的标签
        # 这样还能继续复用你已有的 _select_history_dates，不用改它的注释和实现
        pred_shifted = None
        if hiking_pred_by_day is not None:
            pred_shifted = hiking_pred_by_day.shift(-horizon, freq="D")

        # 循环的“基准日”范围需要预留 horizon 天，避免 target_day 越界
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)
        base_days = pd.date_range(start=start, end=end - pd.Timedelta(days=horizon), freq="D")  # >>> 新增(Dn)

        curves_record: Dict[str, Dict[str, Any]] = {}

        # 将原本按“date”循环，改为按“base_day”循环；其它尽量不动
        for base_day in base_days:
            base_day = base_day.normalize()
            target_day = (base_day + pd.Timedelta(days=horizon)).normalize()  # >>> 新增(Dn)
            date_str = target_day.strftime("%Y-%m-%d")                        # >>> 新增(Dn)：curves_record 的 key=目标日

            # 当天数据（此处的“当天”= 目标日；原注释保持）
            try:
                lr_da_today = self.market_data.loc[date_str, self.col["loadrate_da"]].values
                lr_rt_today = self.market_data.loc[date_str, self.col["loadrate_rt"]].values
                price_da_today = self.market_data.loc[date_str, self.col["price_da"]].values
                price_rt_today = self.market_data.loc[date_str, self.col["price_rt"]].values
            except KeyError:
                continue

            # baseline 之外的模式，需要“目标日”的预测标签（用 pred_shifted 在 base_day 取）
            if selection != "baseline":
                if pred_shifted is None or base_day not in pred_shifted.index:
                    continue

            # ——选择历史日（DA 与 RT 分别按各自真实标签筛选）
            # 复用你原来的工具：anchor_date=base_day，pred_label_by_day=pred_shifted（其在 base_day 上就是 target_day 的标签）
            hist_dates_da = self._select_history_dates(
                all_dates=all_dates,
                anchor_date=base_day,            # >>> 新增(Dn)：锚定为“基准日”
                d=d,
                selection=selection,
                pred_label_by_day=pred_shifted,  # 在 base_day 读到的是 target_day 的 Pred_Label
                true_label_by_day=hiking_truth_da_by_day,
            )
            hist_dates_rt = self._select_history_dates(
                all_dates=all_dates,
                anchor_date=base_day,
                d=d,
                selection=selection,
                pred_label_by_day=pred_shifted,
                true_label_by_day=hiking_truth_rt_by_day if hiking_truth_rt_by_day is not None else hiking_truth_da_by_day,
            )

            # ——拼历史窗口并拟合（与原逻辑一致）
            lr_da_hist = self._concat_days_list(hist_dates_da, self.col["loadrate_da"])
            pr_da_hist = self._concat_days_list(hist_dates_da, self.col["price_da"])
            curve_da_bin, curve_da_interp = self._fit_curve_from_history(lr_da_hist, pr_da_hist)

            lr_rt_hist = self._concat_days_list(hist_dates_rt, self.col["loadrate_rt"])
            pr_rt_hist = self._concat_days_list(hist_dates_rt, self.col["price_rt"])
            curve_rt_bin, curve_rt_interp = self._fit_curve_from_history(lr_rt_hist, pr_rt_hist)

            # ——回代目标日负荷率得到价格预测（保持原注释与分支）
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

            # ——组织与原代码一致的字典（保留原注释）；仅补充 3 个诊断字段
            curves = {
                "bidspace_da": self.market_data.loc[date_str, self.col["bidspace_da"]].values
                    if self.col["bidspace_da"] in self.market_data.columns else None,
                "bidspace_rt": self.market_data.loc[date_str, self.col["bidspace_rt"]].values
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

                # ——最重要：用“历史曲线”对“今天”负荷率的预测——（此处“今天”=目标日）
                "price_pred_da": price_pred_da,
                "price_pred_rt": price_pred_rt,

                # 诊断信息（便于复现/查错）
                "hiking_pred": (int(hiking_pred_by_day.loc[target_day])  # >>> 新增(Dn)：记录目标日的预测标签
                                if (selection != "baseline" and hiking_pred_by_day is not None and target_day in hiking_pred_by_day.index)
                                else None),
                "history_dates_used_da": list(hist_dates_da),
                "history_dates_used_rt": list(hist_dates_rt),
                "selection": selection,

                "horizon": int(horizon),                            # >>> 新增(Dn)
                "base_day": base_day.strftime("%Y-%m-%d"),          # >>> 新增(Dn)
                "target_day": date_str,                             # >>> 新增(Dn)
            }
            curves_record[date_str] = curves

        return curves_record


# ---------- 便捷封装：直接从两个 CSV 构建 curves_record ----------

def build_curves_record_with_history_classify_from_csv(
    market_data: pd.DataFrame,
    date_begin: str,
    date_end: str,
    d: int,
    # 预测CSV（图1）
    pred_csv_path: Optional[str] = None,
    pred_date_col: str = "Date",
    pred_label_col: str = "Pred_Label",
    # 真实CSV（图2）
    truth_csv_path: Optional[str] = None,
    truth_date_col: str = "date",
    truth_da_col: str = "da_hiking",
    truth_rt_col: str = "rt_hiking",
    # 拟合参数
    bin_interval: int = 3,
    max_load_rate: int = 100,
    new_interval: int = 1,
    selection: str = "match",
    horizon: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    一行生成 curves_record（可直接接你的两个画图函数）：
      - selection='baseline'：不需要预测CSV；真实CSV也可不传（不筛标签，纯前 d 天）。
      - 其他 selection：需要真实CSV；'match' 还需要预测CSV。
    """
    # 加载预测与真实标签（按需）
    pred_s = None
    truth_da_s, truth_rt_s = None, None

    if selection != "baseline":
        if truth_csv_path is None:
            raise ValueError("selection!='baseline' 时必须提供真实CSV。")
        truth_da_s, truth_rt_s = load_truth_series_from_csv(
            truth_csv_path, date_col=truth_date_col, da_col=truth_da_col, rt_col=truth_rt_col
        )
        if selection == "match":
            if pred_csv_path is None:
                raise ValueError("selection='match' 需要提供预测CSV。")
            pred_s = load_pred_series_from_csv(pred_csv_path, date_col=pred_date_col, pred_label_col=pred_label_col)

    forecaster = RollingLoadratePriceForecaster_classify(
        market_data=market_data,
        bin_interval=bin_interval,
        max_load_rate=max_load_rate,
        new_interval=new_interval,
    )
    return forecaster.fit_predict(
        start_date=date_begin,
        end_date=date_end,
        d=d,
        hiking_pred_by_day=pred_s,
        hiking_truth_da_by_day=truth_da_s,
        hiking_truth_rt_by_day=truth_rt_s,
        selection=selection,
        horizon=horizon,
    )


# ---------- 便捷封装：直接用 Series（若你在外面已读好 CSV） ----------

def build_curves_record_with_history_classify(
    market_data: pd.DataFrame,
    date_begin: str,
    date_end: str,
    d: int,
    hiking_pred: Optional[Union[pd.Series, Dict[str, int]]] = None,
    hiking_truth_da: Optional[Union[pd.Series, Dict[str, int]]] = None,
    hiking_truth_rt: Optional[Union[pd.Series, Dict[str, int]]] = None,
    bin_interval: int = 3,
    max_load_rate: int = 100,
    new_interval: int = 1,
    selection: str = "match",
    horizon: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    若你已在外部把 CSV 读成 Series，可用这个封装。
    """
    def _norm(s):
        if s is None:
            return None
        if isinstance(s, dict):
            s = pd.Series(s)
        s.index = pd.to_datetime(s.index).normalize()
        s = s.sort_index()
        return s.astype(int)

    pred_s = _norm(hiking_pred)
    truth_da_s = _norm(hiking_truth_da)
    truth_rt_s = _norm(hiking_truth_rt) if hiking_truth_rt is not None else None

    forecaster = RollingLoadratePriceForecaster_classify(
        market_data=market_data,
        bin_interval=bin_interval,
        max_load_rate=max_load_rate,
        new_interval=new_interval,
    )
    return forecaster.fit_predict(
        start_date=date_begin,
        end_date=date_end,
        d=d,
        hiking_pred_by_day=pred_s,
        hiking_truth_da_by_day=truth_da_s,
        hiking_truth_rt_by_day=truth_rt_s,
        selection=selection,
        horizon=horizon,
    )


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



# ====== 便捷函数：从预测结果CSV中分离出“预测正确/错误”的日期列表 ======
import pandas as pd
import numpy as np

def split_correct_wrong_by_csv(
    csv_path: str,
    date_begin: str,
    date_end: str,
    date_col: str = "Date",
    true_col: str = "True_Label",
    pred_col: str = "Pred_Label",
):
    """
    读取预测结果CSV（含 True_Label/Pred_Label），过滤 [date_begin, date_end] 区间，
    返回两个列表：预测正确的日期、预测错误的日期（均为 'YYYY-MM-DD' 字符串）。

    参数
    ----
    csv_path : 预测结果CSV路径（形如 multi_GRU_predict_results_V2.0.csv）
    date_begin, date_end : 起止日期（'YYYY-MM-DD'）
    date_col : 日期列名（默认 'Date'）
    true_col : 真实标签列名（默认 'True_Label'）
    pred_col : 预测标签列名（默认 'Pred_Label'）

    返回
    ----
    correct_dates, wrong_dates : List[str], List[str]
    """

    def _to01_series(s: pd.Series) -> pd.Series:
        def conv(v):
            if pd.isna(v): return np.nan
            if isinstance(v, (bool, np.bool_)): return int(v)
            sv = str(v).strip().lower()
            if sv in ("true", "t", "yes", "y"): return 1
            if sv in ("false", "f", "no", "n"): return 0
            try:
                return int(round(float(sv)))  # 兼容 '1'/'0'/'1.0'
            except Exception:
                return np.nan
        return s.map(conv)

    df = pd.read_csv(csv_path)

    # 解析日期 & 过滤区间
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    begin = pd.to_datetime(date_begin)
    end = pd.to_datetime(date_end)
    df = df[(df[date_col] >= begin) & (df[date_col] <= end)].copy()

    # 若同一日期多行，保留最后一行
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    # 规范标签到 {0,1}
    df["__true"] = _to01_series(df[true_col]).astype("Int64")
    df["__pred"] = _to01_series(df[pred_col]).astype("Int64")
    df = df.dropna(subset=["__true", "__pred"])

    correct_mask = df["__true"] == df["__pred"]
    correct_dates = df.loc[correct_mask, date_col].dt.strftime("%Y-%m-%d").tolist()
    wrong_dates   = df.loc[~correct_mask, date_col].dt.strftime("%Y-%m-%d").tolist()

    return correct_dates, wrong_dates
