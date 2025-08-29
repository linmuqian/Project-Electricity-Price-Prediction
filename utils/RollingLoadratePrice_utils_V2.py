# -*- coding: utf-8 -*-
"""
RollingLoadratePrice_utils_V2.py

【目标】
- 与 V1 完全兼容，同时统一接口与命名。
- 通过 `pattern` 选择运行模式："backtest"（回测）或 "inference"（推理）。
- 通过 `x_mode` 选择自变量："loadrate"（负荷率）或 "bidspace"（竞价空间）。
- 尽量不改原有注释；新增/修改均标注 `# >>> 新增(V2)` / `# >>> 修改(V2)`。

【兼容性】
- 当 `pattern="backtest"` 且 `x_mode="loadrate"` 时，完全委托 V1 的
  `RollingLoadratePriceForecaster_classify.fit_predict`，返回结构与 V1 一致。
- 其余组合（包括推理 & 竞价空间）由本文件实现，同时输出与 V1 相同的关键键：
  `price_pred_da/price_pred_rt` 等；另外**新增** `bidspace_*` 曲线便于诊断。

【依赖】
- 依赖 V1 文件：RollingLoadratePrice_utils_V1.py（位于同一包目录下）。
- 依赖工具：ElecPriceCurve_utils.py、date_utils.py（与 V1 相同）。
"""
from typing import Dict, Any, Optional, Union, Tuple, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ——复用你现有的工具与 V1 能力（接口保持不变）——
from .ElecPriceCurve_utils import (
    find_y_vectorized,
    bin_curve_model,              # V1 的“负荷率→价格”分箱
    interpolate_bin_curve,        # V1 的插值至 1% 网格
)
from .date_utils import find_decade_dates
from .RollingLoadratePrice_utils_V1 import (
    RollingLoadratePriceForecaster_classify as _V1Forecaster,
    load_pred_series_from_csv,
    load_truth_series_from_csv,
    process_market_data,
    plot_compare_predict_vs_true,
    plot_loadrate_price_curves,
)

import matplotlib.font_manager as fm
from matplotlib import rcParams

# myfont = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
# rcParams['font.sans-serif'] = [myfont.get_name()]
# rcParams['axes.unicode_minus'] = False

from matplotlib import rcParams, font_manager as fm

try:
    fams = {f.name for f in fm.fontManager.ttflist}
    order = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
    rcParams["font.sans-serif"] = [f for f in order if f in fams] or ["DejaVu Sans"]
except Exception:
    pass

rcParams["axes.unicode_minus"] = False


# ======================================================================
# >>> 新增(V2)：通用的“连续自变量 → 价格”分箱与插值（适用于竞价空间等）
# ======================================================================

def _midpoint(iv: pd.Interval) -> float:
    return (float(iv.left) + float(iv.right)) / 2.0


def bin_curve_model_generic(
    x: pd.Series,
    y: pd.Series,
    *,
    method: str = "quantile",   # 'quantile' | 'fixed'
    n_bins: int = 30,            # quantile 的分箱数
    fixed_bin_width: Optional[float] = None,  # fixed 时的箱宽(MW)
    right: bool = False,
) -> pd.Series:
    """
    【新增(V2)】将任意连续自变量 x（如竞价空间）与 y=价格 做分箱取均值，获得“静态量价曲线”。
    - method='quantile'（默认）：用 qcut 做等频分箱（样本分布不均匀时更稳健）。
    - method='fixed'：用固定箱宽（需提供 fixed_bin_width）。

    返回：pd.Series，索引为 pd.IntervalIndex，值为均值价格（已按箱区间从小到大排序）。
    """
    df = pd.DataFrame({"x": x.copy(), "y": y.copy()}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    if x_min == x_max:
        # 所有 x 相同，无法分箱；退化为单点区间
        iv = pd.Interval(left=x_min - 1e-6, right=x_max + 1e-6, closed=('right' if right else 'left'))
        s = pd.Series(index=pd.IntervalIndex([iv]), data=[float(df["y"].mean())])
        return s

    if method == "quantile":
        # qcut 可能因大量重复值导致箱边界相同，此处做保护性降维
        for nb in range(min(n_bins, 60), 1, -1):
            try:
                cats = pd.qcut(df["x"], q=nb, duplicates="drop")
                by_bin = df.groupby(cats, observed=False)["y"].mean()
                # 排序（qcut 的 IntervalIndex 已有序，此处显式排序以稳妥）
                by_bin = by_bin.sort_index(key=lambda idx: [iv.left for iv in idx])
                return by_bin
            except Exception:
                continue
        # 若 qcut 彻底失败，退回 fixed
        method = "fixed"

    if method == "fixed":
        if not fixed_bin_width or fixed_bin_width <= 0:
            # 自动给一个箱宽（30 等分）
            fixed_bin_width = (x_max - x_min) / float(max(n_bins, 2))
        # 生成边界并 cut
        edges = np.arange(x_min, x_max + fixed_bin_width, fixed_bin_width)
        edges[-1] = x_max + 1e-9  # 浮点保护
        cats = pd.cut(df["x"], bins=edges, right=right)
        by_bin = df.groupby(cats, observed=False)["y"].mean().sort_index(key=lambda idx: [iv.left for iv in idx])
        return by_bin

    raise ValueError("method must be 'quantile' or 'fixed'.")


def interpolate_curve_from_bins(
    price_mean_by_bin: pd.Series,
    *,
    grid_step: Optional[float] = None,  # 为 None 则自动 100 等分
    kind: str = "quadratic",            # 与 V1 保持一致，默认二次插值
    clip: Tuple[float, float] = (0.0, 1500.0),
) -> Optional[pd.Series]:
    """
    【新增(V2)】将“分箱均值曲线(IntervalIndex)”插值为均匀网格（NumericIndex + IntervalIndex 包裹）。
    返回：pd.Series，索引为 pd.IntervalIndex（等步长网格），值为插值后的价格。
    """
    if price_mean_by_bin is None or price_mean_by_bin.empty:
        return None
    idx: pd.IntervalIndex = pd.IntervalIndex(price_mean_by_bin.index)
    x_mid = np.array([_midpoint(iv) for iv in idx], dtype=float)
    y_val = price_mean_by_bin.values.astype(float)

    _kind = kind if len(x_mid) >= 3 else "linear"  # 点数少降级
    f = interp1d(x_mid, y_val, kind=_kind, fill_value="extrapolate")

    x_min, x_max = float(np.nanmin(x_mid)), float(np.nanmax(x_mid))
    if grid_step is None or grid_step <= 0:
        grid_step = (x_max - x_min) / 100.0 if x_max > x_min else 1.0
    grid = np.arange(x_min, x_max + grid_step, grid_step)

    new_y = f(grid)
    if clip is not None:
        lo, hi = clip
        new_y = np.clip(new_y, lo, hi)

    half = grid_step / 2.0
    new_intervals = [pd.Interval(g - half, g + half, closed='left') for g in grid]
    return pd.Series(new_y, index=pd.IntervalIndex(new_intervals))


def eval_curve_on_x(
    x_list: Sequence[float],
    curve_interp: Optional[pd.Series] = None,
    price_mean_by_bin: Optional[pd.Series] = None,
) -> pd.Series:
    """
    【新增(V2)】在给定 x_list 上评估曲线：优先用插值曲线；若无则退回分箱曲线的“区间取值”。
    复用 V1 的 find_y_vectorized（要求索引为 IntervalIndex）。
    """
    if curve_interp is not None:
        idx: pd.IntervalIndex = pd.IntervalIndex(curve_interp.index)
        x_mid = np.array([_midpoint(iv) for iv in idx], dtype=float)
        y_val = curve_interp.values.astype(float)
        kind = "quadratic" if len(x_mid) >= 3 else "linear"
        # 如果越界，则对最外侧做线性/二次外推，再截取到(0,1500)
        # f = interp1d(x_mid, y_val, kind=kind, fill_value="extrapolate")
        # 如果越界，则贴左右端点，这样更加可控
        f = interp1d(x_mid, y_val, kind=kind, bounds_error=False, fill_value=(y_val[0], y_val[-1]))
        y = f(np.array(x_list, dtype=float))
        y = np.clip(y, 0.0, 1500.0)
        return pd.Series(y, index=pd.Index(x_list))
    elif price_mean_by_bin is not None and isinstance(price_mean_by_bin.index, pd.IntervalIndex):
        return find_y_vectorized(list(x_list), price_mean_by_bin)
    else:
        return pd.Series([np.nan] * len(x_list), index=pd.Index(x_list))


# ======================================================================
# >>> 新增(V2)：统一的 Forecaster（pattern/x_mode 可选）
# ======================================================================
class RollingLoadratePriceForecaster:
    """
    统一版 Forecaster。

    参数
    ----
    market_data : pd.DataFrame      已对齐的市场数据（索引为时间戳）。
    col_map     : Dict[str,str]     列名映射（见默认映射）。
    require_min_bins : int          插值所需的最少有效分箱点数。

    默认列名映射（与 V1 对齐）：
      loadrate_da → "日前负荷率"
      loadrate_rt → "实时负荷率"
      price_da    → "日前价格"   （V1: market_data["price"]）
      price_rt    → "实时价格"
      bidspace_da → "日前竞价空间"
      bidspace_rt → "实时竞价空间"
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        col_map: Optional[Dict[str, str]] = None,
        require_min_bins: int = 3,
        bin_interval: int = 3,      # 仅 x_mode=loadrate 时使用（与 V1 一致）
        max_load_rate: int = 100,
        new_interval: int = 1,
    ) -> None:
        self.market_data = market_data.sort_index()
        default_map = {
            "loadrate_da": "日前负荷率",
            "loadrate_rt": "实时负荷率",
            "price_da": "日前价格",
            "price_rt": "实时价格",
            "bidspace_da": "日前竞价空间",
            "bidspace_rt": "实时竞价空间",
        }
        self.col = default_map if col_map is None else {**default_map, **col_map}
        self.require_min_bins = require_min_bins
        self.bin_interval = bin_interval
        self.max_load_rate = max_load_rate
        self.new_interval = new_interval

    # -------------------- 主流程（统一接口） --------------------
    def fit_predict(
        self,
        start_date: str,
        end_date: str,
        d: int,
        *,
        selection: str = "match",               # 与 V1 相同
        pattern: str = "backtest",              # "backtest" | "inference"
        x_mode: str = "loadrate",               # "loadrate" | "bidspace"
        # 标签（与 V1 相同语义）
        hiking_pred_by_day: Optional[pd.Series] = None,
        hiking_truth_da_by_day: Optional[pd.Series] = None,
        hiking_truth_rt_by_day: Optional[pd.Series] = None,
        # x_mode=bidspace 的分箱/插值参数
        bin_method: str = "quantile",
        n_bins: int = 30,
        fixed_bin_width: Optional[float] = None,
        grid_step: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        统一的 fit_predict。
        - 当 (pattern, x_mode) == ("backtest","loadrate")：直接调用 V1 代码，完全兼容。
        - 其他组合：使用本文件的通用实现。
        返回：`curves_record`，与 V1 的键保持一致；另有 bidspace_* 供诊断。
        """
        # ——回测模式（直接委托 V1）——
        # 量价关系是【负荷率】->【价格】
        if pattern == "backtest" and x_mode == "loadrate":
            fore = _V1Forecaster(
                market_data=self.market_data,
                bin_interval=self.bin_interval,
                max_load_rate=self.max_load_rate,
                new_interval=self.new_interval,
            )
            return fore.fit_predict(
                start_date=start_date,
                end_date=end_date,
                d=d,
                hiking_pred_by_day=hiking_pred_by_day,
                hiking_truth_da_by_day=hiking_truth_da_by_day,
                hiking_truth_rt_by_day=hiking_truth_rt_by_day,
                selection=selection,
            )
        
        # ——推理模式（或其它组合）——
        # 量价关系主要是【竞价空间】->【价格】
        # ——其余组合：统一实现（x 可为 loadrate 或 bidspace；pattern 可为 backtest 或 inference）——
        def _norm(s: Optional[pd.Series]) -> Optional[pd.Series]:
            if s is None:
                return None
            s = s.copy()
            s.index = pd.to_datetime(s.index).normalize()
            return s.sort_index()

        hiking_pred_by_day = _norm(hiking_pred_by_day)
        hiking_truth_da_by_day = _norm(hiking_truth_da_by_day)
        hiking_truth_rt_by_day = _norm(hiking_truth_rt_by_day)

        # 列选择（x 的来源）
        if x_mode == "loadrate":
            x_col_da = self.col["loadrate_da"]
            x_col_rt = self.col["loadrate_rt"]
        elif x_mode == "bidspace":
            x_col_da = self.col["bidspace_da"]
            x_col_rt = self.col["bidspace_rt"]
        else:
            raise ValueError("x_mode must be 'loadrate' or 'bidspace'.")

        price_da_col = self.col["price_da"]
        price_rt_col = self.col["price_rt"]

        all_dates = pd.to_datetime(self.market_data.index.normalize().unique())
        curves_record: Dict[str, Dict[str, Any]] = {}

        # ——历史日选择（与 V1 语义一致）
        def _select_history_dates(anchor_date: pd.Timestamp, truth_by_day: Optional[pd.Series]) -> np.ndarray:
            anchor_day = anchor_date.normalize()
            prev_dates = all_dates[all_dates < anchor_day]
            if selection == "baseline":
                candidates = prev_dates
            else:
                if truth_by_day is None:
                    return np.array([], dtype=str)
                if selection == "match":
                    if hiking_pred_by_day is None or anchor_day not in hiking_pred_by_day.index:
                        return np.array([], dtype=str)
                    target = int(hiking_pred_by_day.loc[anchor_day])
                elif selection == "pos_only":
                    target = 1
                elif selection == "neg_only":
                    target = 0
                else:
                    raise ValueError("Unknown selection (use 'baseline'|'match'|'pos_only'|'neg_only').")
                prev_truth = truth_by_day.reindex(prev_dates)
                mask = (prev_truth == target).fillna(False).values
                candidates = prev_dates[mask]
            if candidates.size == 0:
                return np.array([], dtype=str)
            selected = candidates[-d:]
            return np.array([pd.Timestamp(x).strftime('%Y-%m-%d') for x in selected])

        # ——拼历史窗口——
        def _concat_days(days: np.ndarray, col: str) -> np.ndarray:
            arrs = []
            for ds in days:
                try:
                    s = self.market_data.loc[ds, col]
                except KeyError:
                    continue
                vals = s.values if hasattr(s, 'values') else np.array([s])
                vals = vals[~pd.isna(vals)]
                if vals.size:
                    arrs.append(vals)
            return np.concatenate(arrs) if arrs else np.array([])

        # ——拟合曲线（根据 x_mode 选择分箱/插值方案）——
        def _fit_curve(x_hist: np.ndarray, y_hist: np.ndarray):
            if x_hist.size == 0 or y_hist.size == 0:
                return None, None
            if x_mode == "loadrate":
                by_bin = bin_curve_model(
                    load_rate=pd.Series(x_hist),
                    price=pd.Series(y_hist),
                    bin_interval=self.bin_interval,
                    max_load_rate=self.max_load_rate,
                )
                if by_bin.dropna().shape[0] < self.require_min_bins:
                    return by_bin, None
                curve_interp = interpolate_bin_curve(by_bin, new_interval=self.new_interval)
                return by_bin, curve_interp
            else:  # bidspace
                by_bin = bin_curve_model_generic(pd.Series(x_hist), pd.Series(y_hist),
                                                 method=bin_method, n_bins=n_bins, fixed_bin_width=fixed_bin_width)
                if by_bin.dropna().shape[0] < self.require_min_bins:
                    return by_bin, None
                curve_interp = interpolate_curve_from_bins(by_bin, grid_step=grid_step)
                return by_bin, curve_interp

        # ——主循环：按日处理（backtest 与 inference 的区别仅在于“当天 x 是否要求存在真实价格”）——
        for date in pd.date_range(start=start_date, end=end_date, freq="D"):
            dstr = date.strftime('%Y-%m-%d')

            # 当天 x（回代所需）；若缺失则跳过该日
            try:
                x_da_today = self.market_data.loc[dstr, x_col_da].values
            except KeyError:
                continue
            x_rt_today = None
            if x_col_rt in self.market_data.columns:
                try:
                    x_rt_today = self.market_data.loc[dstr, x_col_rt].values
                except KeyError:
                    x_rt_today = None

            # 除 baseline 外，需要当天有预测标签（与 V1 一致）
            if selection != "baseline":
                if hiking_pred_by_day is None or date.normalize() not in hiking_pred_by_day.index:
                    continue

            # 历史窗口（DA/RT 各自）
            hist_da = _select_history_dates(date, hiking_truth_da_by_day)
            hist_rt = _select_history_dates(date, hiking_truth_rt_by_day if hiking_truth_rt_by_day is not None else hiking_truth_da_by_day)

            # 历史 x/y（价格列恒为 price_*）
            x_da_hist = _concat_days(hist_da, x_col_da)
            y_da_hist = _concat_days(hist_da, price_da_col)
            x_rt_hist = _concat_days(hist_rt, x_col_rt) if x_col_rt in self.market_data.columns else np.array([])
            y_rt_hist = _concat_days(hist_rt, price_rt_col)

            # 拟合曲线
            curve_da_bin, curve_da_interp = _fit_curve(x_da_hist, y_da_hist)
            curve_rt_bin, curve_rt_interp = _fit_curve(x_rt_hist, y_rt_hist) if x_rt_hist.size else (None, None)

            # 回代当天 x → 预测价格
            price_pred_da = eval_curve_on_x(x_da_today, curve_da_interp, curve_da_bin)
            if x_rt_today is not None and len(x_rt_today) > 0:
                price_pred_rt = eval_curve_on_x(x_rt_today, curve_rt_interp, curve_rt_bin)
            else:
                price_pred_rt = pd.Series([np.nan] * len(x_da_today))

            # 真实价格（回测通常可得；推理日可能 NaN）
            try:
                price_da_today = self.market_data.loc[dstr, price_da_col].values
            except Exception:
                price_da_today = np.array([np.nan] * len(price_pred_da))
            try:
                price_rt_today = self.market_data.loc[dstr, price_rt_col].values
            except Exception:
                price_rt_today = np.array([np.nan] * len(price_pred_rt))

            # ——输出结构与 V1 保持一致（额外输出 bidspace_* 以便诊断）——
            curves: Dict[str, Any] = {
                "bidspace_da": (x_da_today if x_mode == "bidspace" else None),
                "bidspace_rt": (x_rt_today if x_mode == "bidspace" else None),
                "loadrate_da": (x_da_today if x_mode == "loadrate" else None),
                "loadrate_rt": (x_rt_today if x_mode == "loadrate" else None),

                "price_da": price_da_today,
                "price_rt": price_rt_today,

                # 历史拟合得到的静态曲线（按两种 x 分别放置）
                "loadrate_price_bin_curve_da": (curve_da_bin if x_mode == "loadrate" else None),
                "loadrate_price_bin_curve_rt": (curve_rt_bin if x_mode == "loadrate" else None),
                "loadrate_price_bin_curve_interpolate_da": (curve_da_interp if x_mode == "loadrate" else None),
                "loadrate_price_bin_curve_interpolate_rt": (curve_rt_interp if x_mode == "loadrate" else None),

                "bidspace_price_bin_curve_da": (curve_da_bin if x_mode == "bidspace" else None),
                "bidspace_price_bin_curve_rt": (curve_rt_bin if x_mode == "bidspace" else None),
                "bidspace_price_curve_interpolate_da": (curve_da_interp if x_mode == "bidspace" else None),
                "bidspace_price_curve_interpolate_rt": (curve_rt_interp if x_mode == "bidspace" else None),

                # ——最重要：预测序列（键名与 V1 一致）——
                "price_pred_da": price_pred_da,
                "price_pred_rt": price_pred_rt,

                # 诊断
                "hiking_pred": (int(hiking_pred_by_day.loc[date.normalize()])
                                  if (selection != "baseline" and hiking_pred_by_day is not None and date.normalize() in hiking_pred_by_day.index)
                                  else None),
                "history_dates_used_da": list(hist_da),
                "history_dates_used_rt": list(hist_rt),
                "selection": selection,
                "pattern": pattern,
                "x_mode": x_mode,
            }
            curves_record[dstr] = curves

        return curves_record


# ======================================================================
# >>> 新增/统一：一站式构建（直接从 CSV）
# ======================================================================

def build_curves_record_with_history_from_csv(
    market_data: pd.DataFrame,
    date_begin: str,
    date_end: str,
    d: int,
    *,
    pattern: str = "backtest",           # "backtest" | "inference"
    x_mode: str = "loadrate",            # "loadrate" | "bidspace"
    col_map: Optional[Dict[str, str]] = None,
    # 预测CSV，即预测是否hiking=0或者1
    pred_csv_path: Optional[str] = None,
    pred_date_col: str = "Date",
    pred_label_col: str = "Pred_Label",
    # 真实CSV，即真实的前几日的历史hiking数据
    truth_csv_path: Optional[str] = None,
    truth_date_col: str = "date",
    truth_da_col: str = "da_hiking",
    truth_rt_col: str = "rt_hiking",
    # selection 与分箱参数
    selection: str = "match",
    bin_method: str = "quantile",
    n_bins: int = 30,
    fixed_bin_width: Optional[float] = None,
    grid_step: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    统一的一键构建：直接从两个 CSV + 市场数据，得到 curves_record。
    - pattern/x_mode 控制四象限逻辑；
    - 'backtest+loadrate' 走 V1；其它组合走 V2 通用实现。
    """
    pred_s = None
    truth_da_s = None
    truth_rt_s = None

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

    fore = RollingLoadratePriceForecaster(market_data=market_data, col_map=col_map)
    return fore.fit_predict(
        start_date=date_begin,
        end_date=date_end,
        d=d,
        selection=selection,
        pattern=pattern,
        x_mode=x_mode,
        hiking_pred_by_day=pred_s,
        hiking_truth_da_by_day=truth_da_s,
        hiking_truth_rt_by_day=truth_rt_s,
        bin_method=bin_method,
        n_bins=n_bins,
        fixed_bin_width=fixed_bin_width,
        grid_step=grid_step,
    )


# ======================================================================
# >>> 新增/统一：一站式构建（若 CSV 已在外部读成 Series/Dict）
# ======================================================================

def build_curves_record_with_history(
    market_data: pd.DataFrame,
    date_begin: str,
    date_end: str,
    d: int,
    *,
    pattern: str = "backtest",
    x_mode: str = "loadrate",
    hiking_pred: Optional[Union[pd.Series, Dict[str, int]]] = None,
    hiking_truth_da: Optional[Union[pd.Series, Dict[str, int]]] = None,
    hiking_truth_rt: Optional[Union[pd.Series, Dict[str, int]]] = None,
    col_map: Optional[Dict[str, str]] = None,
    bin_method: str = "quantile",
    n_bins: int = 30,
    fixed_bin_width: Optional[float] = None,
    grid_step: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """统一封装：若 CSV 已在外部读成 Series/Dict，可用本函数。"""
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

    fore = RollingLoadratePriceForecaster(market_data=market_data, col_map=col_map)
    return fore.fit_predict(
        start_date=date_begin,
        end_date=date_end,
        d=d,
        selection="match" if pred_s is not None else "baseline",
        pattern=pattern,
        x_mode=x_mode,
        hiking_pred_by_day=pred_s,
        hiking_truth_da_by_day=truth_da_s,
        hiking_truth_rt_by_day=truth_rt_s,
        bin_method=bin_method,
        n_bins=n_bins,
        fixed_bin_width=fixed_bin_width,
        grid_step=grid_step,
    )


# ======================================================================
# >>> 兼容旧名（V1 的函数名保持可用）
# ======================================================================

def build_curves_record_with_history_classify_from_csv(*args, **kwargs):
    """兼容旧接口：等价于 build_curves_record_with_history_from_csv(pattern='backtest', x_mode='loadrate')."""
    kwargs.setdefault('pattern', 'backtest')
    kwargs.setdefault('x_mode', 'loadrate')
    return build_curves_record_with_history_from_csv(*args, **kwargs)


def build_curves_record_with_history_classify(*args, **kwargs):
    """兼容旧接口：等价于 build_curves_record_with_history(pattern='backtest', x_mode='loadrate')."""
    kwargs.setdefault('pattern', 'backtest')
    kwargs.setdefault('x_mode', 'loadrate')
    return build_curves_record_with_history(*args, **kwargs)


# ======================================================================
# >>> 作图辅助（新增：竞价空间；并做空结果保护）
# ======================================================================

def plot_bidspace_price_curves(curves_record: Dict[str, Dict[str, Any]]):
    """
    【新增(V2)】绘制“价格-竞价空间”的静态曲线（DA/RT），便于检查拟合是否合理。
    与 V1 的 plot_loadrate_price_curves 类似，但横轴改为竞价空间(MW)。
    """
    if not curves_record:
        print("【提示】curves_record 为空：可能推理日缺少'日前竞价空间/价格'数据，或历史窗口/标签筛选后为空。")
        return

    import math
    n_cols = 6
    n_rows = max(1, math.ceil(len(curves_record) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (date_str, curves) in enumerate(curves_record.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        curve_da = curves.get("bidspace_price_curve_interpolate_da")
        curve_rt = curves.get("bidspace_price_curve_interpolate_rt")

        if isinstance(curve_da, pd.Series) and isinstance(curve_da.index, pd.IntervalIndex):
            x_da = np.fromiter((iv.left for iv in curve_da.index), dtype=float)
            ax.plot(x_da, curve_da.values, label="日前(竞价空间→价格)", linestyle='-', alpha=0.9)
        else:
            ax.text(0.5, 0.55, "历史不足，无法拟合DA", transform=ax.transAxes, ha="center", va="center")

        if isinstance(curve_rt, pd.Series) and isinstance(curve_rt.index, pd.IntervalIndex):
            x_rt = np.fromiter((iv.left for iv in curve_rt.index), dtype=float)
            ax.plot(x_rt, curve_rt.values, label="实时(竞价空间→价格)", linestyle='--', alpha=0.9)
        else:
            ax.text(0.5, 0.40, "历史不足，无法拟合RT", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(f"{date_str}")
        ax.set_xlabel("竞价空间(MW)")
        ax.set_ylabel("价格")
        ax.set_ylim(0, 1400)
        ax.legend()
        ax.grid(True)

    total = n_rows * n_cols
    for i in range(len(curves_record), total):
        r, c = divmod(i, n_cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


def plot_inference_price_timeseries(curves_record: Dict[str, Dict[str, Any]]):
    """
    【新增(V2)】推理日的“预测价格(日前/实时) 时序图”。
    - 若真实价格不可得（推理日），对比曲线仅显示预测。
    """
    if not curves_record:
        print("【提示】curves_record 为空：请检查推理日是否有可用数据/标签。")
        return

    import math
    n_cols = 6
    n_rows = max(1, math.ceil(len(curves_record) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), sharex=False)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (date_str, curves) in enumerate(curves_record.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        pred_da = curves.get("price_pred_da")
        true_da = curves.get("price_da")
        pred_rt = curves.get("price_pred_rt")
        true_rt = curves.get("price_rt")

        if isinstance(pred_da, (pd.Series, np.ndarray)):
            ax.plot(np.asarray(pred_da), label="预测-日前", linestyle='--')
        if isinstance(true_da, (pd.Series, np.ndarray)) and not (np.isnan(true_da).all() if isinstance(true_da, np.ndarray) else pd.isna(true_da).all()):
            ax.plot(np.asarray(true_da), label="真实-日前", alpha=0.5)

        if isinstance(pred_rt, (pd.Series, np.ndarray)):
            ax.plot(np.asarray(pred_rt), label="预测-实时", linestyle='--')
        if isinstance(true_rt, (pd.Series, np.ndarray)) and not (np.isnan(true_rt).all() if isinstance(true_rt, np.ndarray) else pd.isna(true_rt).all()):
            ax.plot(np.asarray(true_rt), label="真实-实时", alpha=0.5)

        ax.set_title(f"{date_str} 推理预测")
        ax.set_ylabel("价格")
        ax.legend()
        ax.grid(True)

    total = n_rows * n_cols
    for i in range(len(curves_record), total):
        r, c = divmod(i, n_cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()


# ======================================================================
# >>> 新增(V2)：统一的绘图入口（按 pattern/x_mode 分流）
# ======================================================================

def _first_item(curves_record: Dict[str, Dict[str, Any]]):
    """# >>> 新增(V2) 取第一天数据（推理模式通常只画一天）。"""
    if not curves_record:
        return None, None
    # Python 3.7+ dict保持插入顺序
    for k, v in curves_record.items():
        return k, v
    return None, None


def _plot_single_bidspace_curve(date_str: str, curves: Dict[str, Any]):
    """# >>> 新增(V2) 推理模式：单图绘制“价格-竞价空间”静态曲线。"""
    if curves is None:
        print("【提示】无可绘数据。")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

    curve_da = curves.get("bidspace_price_curve_interpolate_da")
    curve_rt = curves.get("bidspace_price_curve_interpolate_rt")

    if isinstance(curve_da, pd.Series) and isinstance(curve_da.index, pd.IntervalIndex):
        x_da = np.fromiter((iv.left for iv in curve_da.index), dtype=float)
        ax.plot(x_da, curve_da.values, label="日前(竞价空间→价格)", linestyle='-')
    else:
        ax.text(0.5, 0.55, "历史不足，无法拟合DA", transform=ax.transAxes, ha="center", va="center")

    if isinstance(curve_rt, pd.Series) and isinstance(curve_rt.index, pd.IntervalIndex):
        x_rt = np.fromiter((iv.left for iv in curve_rt.index), dtype=float)
        ax.plot(x_rt, curve_rt.values, label="实时(竞价空间→价格)", linestyle='--')
    else:
        ax.text(0.5, 0.40, "历史不足，无法拟合RT", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(f"{date_str}量价关系")
    ax.set_xlabel("竞价空间(MW)")
    ax.set_ylabel("价格")
    ax.set_ylim(0, 1400)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _plot_single_loadrate_curve(date_str: str, curves: Dict[str, Any]):
    """# >>> 新增(V2) 推理模式：单图绘制“价格-负荷率”静态曲线。"""
    if curves is None:
        print("【提示】无可绘数据。")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

    curve_da = curves.get("loadrate_price_bin_curve_interpolate_da")
    curve_rt = curves.get("loadrate_price_bin_curve_interpolate_rt")

    if isinstance(curve_da, pd.Series) and isinstance(curve_da.index, pd.IntervalIndex):
        x_da = np.fromiter((iv.left for iv in curve_da.index), dtype=float)
        ax.plot(x_da, curve_da.values, label="日前量价曲线", linestyle='-')
    else:
        ax.text(0.5, 0.55, "历史不足，无法拟合DA", transform=ax.transAxes, ha="center", va="center")

    if isinstance(curve_rt, pd.Series) and isinstance(curve_rt.index, pd.IntervalIndex):
        x_rt = np.fromiter((iv.left for iv in curve_rt.index), dtype=float)
        ax.plot(x_rt, curve_rt.values, label="实时量价曲线", linestyle='--')
    else:
        ax.text(0.5, 0.40, "历史不足，无法拟合RT", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(f"{date_str}量价关系")
    ax.set_xlabel("负荷率(%)")
    ax.set_ylabel("价格")
    ax.set_ylim(0, 1400)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _plot_single_price_timeseries(date_str: str, curves: Dict[str, Any]):
    """# >>> 新增(V2) 推理模式：单图绘制‘预测 vs 真实’的日内时序。"""
    if curves is None:
        print("【提示】无可绘数据。")
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))

    pred_da = curves.get("price_pred_da")
    true_da = curves.get("price_da")
    pred_rt = curves.get("price_pred_rt")
    true_rt = curves.get("price_rt")

    if isinstance(pred_da, (pd.Series, np.ndarray)):
        ax.plot(np.asarray(pred_da), label="预测-日前", linestyle='--')
    if isinstance(true_da, (pd.Series, np.ndarray)) and not (
        np.isnan(true_da).all() if isinstance(true_da, np.ndarray) else pd.isna(true_da).all()
    ):
        ax.plot(np.asarray(true_da), label="真实-日前", alpha=0.5)

    if isinstance(pred_rt, (pd.Series, np.ndarray)):
        ax.plot(np.asarray(pred_rt), label="预测-实时", linestyle='--')
    if isinstance(true_rt, (pd.Series, np.ndarray)) and not (
        np.isnan(true_rt).all() if isinstance(true_rt, np.ndarray) else pd.isna(true_rt).all()
    ):
        ax.plot(np.asarray(true_rt), label="真实-实时", alpha=0.5)

    ax.set_title(f"{date_str} 价格预测")
    ax.set_ylabel("价格")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_static_curves_unified(curves_record: Dict[str, Dict[str, Any]], *, pattern: str = "backtest", x_mode: str = "loadrate"):
    """
    # >>> 新增(V2)
    统一的“静态量价曲线”入口。
    - pattern='backtest'：沿用原函数
        * x_mode='loadrate' → plot_loadrate_price_curves(curves_record)
        * x_mode='bidspace' → plot_bidspace_price_curves(curves_record)
    - pattern='inference'：只画一个图（通常只推理一天）
        * x_mode='loadrate' → 单图负荷率曲线
        * x_mode='bidspace' → 单图竞价空间曲线
    """
    if not curves_record:
        print("【提示】curves_record 为空，无法绘图。")
        return

    if pattern == "backtest":
        if x_mode == "loadrate":
            return plot_loadrate_price_curves(curves_record)  # 原函数（不改注释）
        elif x_mode == "bidspace":
            return plot_bidspace_price_curves(curves_record)  # 原函数（不改注释）
        else:
            raise ValueError("x_mode must be 'loadrate' or 'bidspace'.")

    # inference → 单图
    date_str, curves = _first_item(curves_record)
    if date_str is None:
        print("【提示】无可绘数据。")
        return
    if x_mode == "loadrate":
        return _plot_single_loadrate_curve(date_str, curves)
    elif x_mode == "bidspace": 
        return _plot_single_bidspace_curve(date_str, curves)
    else:
        raise ValueError("x_mode must be 'loadrate' or 'bidspace'.")


def plot_price_timeseries_unified(curves_record: Dict[str, Dict[str, Any]], *, pattern: str = "backtest"):
    """
    # >>> 新增(V2)
    统一的“预测 vs 真实 时序图”入口。
    - pattern='backtest'：沿用原函数 plot_compare_predict_vs_true（多子图）
    - pattern='inference'：只画一个图（通常只推理一天）
    """
    if not curves_record:
        print("【提示】curves_record 为空，无法绘图。")
        return

    if pattern == "backtest":
        return plot_compare_predict_vs_true(curves_record)  # 原函数（不改注释）

    date_str, curves = _first_item(curves_record)
    if date_str is None:
        print("【提示】无可绘数据。")
        return
    return _plot_single_price_timeseries(date_str, curves)



# ======================================================================
# >>> 使用示例（仅注释，导入时不执行）
# ======================================================================
# fore = RollingLoadratePriceForecaster(market_data)
# # 1) 回测（V1 原样）
# curves_bt = fore.fit_predict(
#     start_date="2025-08-01", end_date="2025-08-10", d=7,
#     selection="match", pattern="backtest", x_mode="loadrate",
#     hiking_pred_by_day=pred_s, hiking_truth_da_by_day=truth_da, hiking_truth_rt_by_day=truth_rt,
# )
#
# # 2) 推理（竞价空间→价格）
# curves_inf = fore.fit_predict(
#     start_date="2025-08-21", end_date="2025-08-21", d=7,
#     selection="match", pattern="inference", x_mode="bidspace",
#     hiking_pred_by_day=pred_s, hiking_truth_da_by_day=truth_da, hiking_truth_rt_by_day=truth_rt,
#     bin_method="quantile", n_bins=30,
# )
