# utils/bidspace_baseline_utils.py
# ------------------------------------------------------------
# 作用：
# 1) 读取 wind / pv / load 的 dayplus 预测 parquet，并对齐到目标日期
# 2) 计算竞价空间 = 负荷 - 风 - 光
# 3) 支持 "shengdiao"：从 market_Dn_data.parquet 读取 D 天的 *_d{h} 列
# 4) 用 baseline（过去 d 天的竞价空间→价格）预测目标日价格，并保存
# ------------------------------------------------------------

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -------------------- 基础 I/O --------------------

def _ensure_datetime_index(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """统一为 DatetimeIndex（升序）。支持 'time' 列或 'index' 列兜底。"""
    out = df.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col])
        out = out.set_index(time_col)
    elif "index" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["index"] = pd.to_datetime(out["index"])
        out = out.set_index("index")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("输入 DataFrame 既没有 'time' 列，也没有 DatetimeIndex（或 'index' 列）。")
    # 去时区，统一 naive
    try:
        if out.index.tz is not None:
            out.index = out.index.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            out.index = out.index.tz_localize(None)
        except Exception:
            pass
    return out.sort_index()


def read_dayplus_parquet(parquet_path: str, target_date: str) -> pd.Series:
    """
    读取类似 data/processed/pv/dayplus_2_ifs.parquet 的文件，
    返回该 'target_date' 对应的 96 个点的预测序列 Series（索引为时间戳）。
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"文件不存在: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="fastparquet")
    df = _ensure_datetime_index(df, time_col="time")
    # 过滤到目标日
    d0 = pd.to_datetime(target_date).normalize()
    d1 = d0 + pd.Timedelta(days=1)
    df_day = df[(df.index >= d0) & (df.index < d1)]
    if df_day.empty:
        raise ValueError(f"{os.path.basename(parquet_path)} 中没有 {target_date} 的记录。")
    # 兼容列名
    for c in ["pred", "value", "y_hat", "forecast"]:
        if c in df_day.columns:
            s = df_day[c].astype(float).rename(c)
            s.index.name = "time"
            return s
    if df_day.shape[1] == 1:
        s = df_day.iloc[:, 0].astype(float)
        s.index.name = "time"
        return s
    raise ValueError(f"{parquet_path} 找不到预测值列（期望 'pred'）。实际列: {list(df_day.columns)}")


# -------------------- 省调(shengdiao) 读取 --------------------

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def read_shengdiao_dn_series(
    kind: str,                 # 'load' | 'wind' | 'pv'
    base_date: str,            # D：基准日（取 D 天全日 96 点）
    horizon: int,              # 2..5
    market_dn_parquet_path: str,
) -> pd.Series:
    """
    从 market_Dn_data.parquet 读取“D 天的 *_d{horizon} 列”，返回一条序列。
      load → 省调负荷-日前(MW)_d{h}
      wind → 日前风电(MW)_d{h}
      pv   → 日前光伏(MW)_d{h}
    """
    if not os.path.exists(market_dn_parquet_path):
        raise FileNotFoundError(f"省调 Dn parquet 不存在: {market_dn_parquet_path}")
    df = pd.read_parquet(market_dn_parquet_path, engine="fastparquet")
    df = _ensure_datetime_index(df, time_col="time" if "time" in df.columns else "index")

    # 切 D 当天
    d0 = pd.to_datetime(base_date).normalize()
    d1 = d0 + pd.Timedelta(days=1)
    df_day = df[(df.index >= d0) & (df.index < d1)]
    if df_day.empty:
        raise ValueError(f"省调 Dn 数据在 {d0.date()} 当天没有记录。")

    h = int(horizon)
    if kind == "load":
        cands = [f"省调负荷-日前(MW)_d{h}", f"日前负荷(MW)_d{h}", f"省调负荷(MW)_d{h}"]
        col = _pick_col(df_day, cands)
        if col is None:
            raise ValueError(f"省调 Dn 缺少负荷 d{h} 列（尝试 {cands}）。")
        name = "load_pred"
    elif kind == "wind":
        cands = [f"日前风电(MW)_d{h}", f"风电-日前(MW)_d{h}"]
        col = _pick_col(df_day, cands)
        if col is None:
            raise ValueError(f"省调 Dn 缺少风电 d{h} 列（尝试 {cands}）。")
        name = "wind_pred"
    elif kind == "pv":
        cands = [f"日前光伏(MW)_d{h}", f"光伏-日前(MW)_d{h}"]
        col = _pick_col(df_day, cands)
        if col is None:
            raise ValueError(f"省调 Dn 缺少光伏 d{h} 列（尝试 {cands}）。")
        name = "pv_pred"
    else:
        raise ValueError("kind must be 'load' | 'wind' | 'pv'")

    s = df_day[col].astype(float).rename(name)
    s.index.name = "time"
    return s


# -------------------- 历史量价曲线（baseline） --------------------

@dataclass
class Curve:
    x_mid: np.ndarray  # 分箱中心（升序）
    y_mean: np.ndarray # 对应均价
    def is_valid(self) -> bool:
        return isinstance(self.x_mid, np.ndarray) and isinstance(self.y_mean, np.ndarray) and len(self.x_mid) >= 2


def load_market_for_history(market_parquet_path: str) -> pd.DataFrame:
    """
    读取全量市场数据 parquet，需要至少包含：
      - 日前价格：['日前价格','price','日前-价格','DA价格','日前价']
      - 日前竞价空间：['竞价空间-日前(MW)', '日前竞价空间', 'bidspace_da']
    若缺少“日前竞价空间”，尝试用 (省调负荷-日前) - (日前风电) - (日前光伏) 计算。
    """
    df = pd.read_parquet(market_parquet_path, engine="fastparquet")
    df = _ensure_datetime_index(df, time_col="time" if "time" in df.columns else "index")
    # 价格列
    price_col = _pick_col(df, ["日前价格", "price", "日前-价格", "DA价格", "日前价"])
    if price_col is None:
        raise ValueError("市场数据缺少 '日前价格' 列。")
    df = df.rename(columns={price_col: "日前价格"})
    # 竞价空间列
    bid_col = _pick_col(df, ["竞价空间-日前(MW)", "日前竞价空间", "bidspace_da"])
    if bid_col is not None:
        df = df.rename(columns={bid_col: "日前竞价空间"})
        return df

    # 计算竞价空间（负荷-风-光）
    load_cands = ["省调负荷-日前(MW)", "日前负荷(MW)", "日前-负荷(MW)"]
    wind_cands = ["日前风电(MW)", "风电-日前(MW)", "日前-风电(MW)"]
    pv_cands   = ["日前光伏(MW)", "光伏-日前(MW)", "日前-光伏(MW)"]
    def pick(cands): 
        return _pick_col(df, cands)
    lc, wc, pc = pick(load_cands), pick(wind_cands), pick(pv_cands)
    if lc and wc and pc:
        df["日前竞价空间"] = df[lc].astype(float) - df[wc].astype(float) - df[pc].astype(float)
        return df
    raise ValueError("市场数据中既没有现成的 '日前竞价空间'，也无法通过(负荷-风-光)推导。")


def _quantile_bin_curve(x: np.ndarray, y: np.ndarray, n_bins: int = 30) -> Curve:
    """对 (x,y) 做等频分箱，取每箱 y 的均值，返回分箱中心与均价。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 8:
        xs = np.unique(x)
        xs.sort()
        if xs.size >= 2:
            mids = xs[:2]
            order = np.argsort(x)
            y_sorted = y[order]
            x_sorted = x[order]
            y_est = np.interp(mids, x_sorted, y_sorted)
            return Curve(x_mid=mids, y_mean=y_est)
        return Curve(x_mid=np.array([]), y_mean=np.array([]))

    qs = np.linspace(0, 1, num=min(n_bins, max(3, x.size // 4)) + 1)
    bins = np.quantile(x, qs)
    bins = np.unique(bins)
    if bins.size < 3:
        xs = np.unique(x)
        xs.sort()
        if xs.size >= 2:
            mids = xs[:2]
            order = np.argsort(x)
            y_sorted = y[order]
            x_sorted = x[order]
            y_est = np.interp(mids, x_sorted, y_sorted)
            return Curve(x_mid=mids, y_mean=y_est)
        return Curve(x_mid=np.array([]), y_mean=np.array([]))

    mids, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x <= hi if hi == bins[-1] else x < hi)
        if not mask.any():
            continue
        mids.append((lo + hi) / 2.0)
        means.append(float(np.mean(y[mask])))
    order = np.argsort(mids)
    return Curve(x_mid=np.array(mids)[order], y_mean=np.array(means)[order])


def predict_from_curve(x_pred: np.ndarray, curve: Curve, clip: Tuple[float, float] = (0.0, 1500.0)) -> np.ndarray:
    """线性插值，越界贴边。"""
    x_pred = np.asarray(x_pred, dtype=float)
    if not curve.is_valid():
        return np.full_like(x_pred, fill_value=np.nan, dtype=float)
    y_pred = np.interp(x_pred, curve.x_mid, curve.y_mean, left=curve.y_mean[0], right=curve.y_mean[-1])
    if clip is not None:
        lo, hi = clip
        y_pred = np.clip(y_pred, lo, hi)
    return y_pred


def build_baseline_curve_da(
    market_df: pd.DataFrame,
    target_date: str,
    d_hist_days: int = 7,
) -> Curve:
    """用目标日前 d 天的 (竞价空间→价格) 拟合静态曲线。"""
    t = pd.to_datetime(target_date).normalize()
    t0 = t - pd.Timedelta(days=d_hist_days)
    hist = market_df[(market_df.index >= t0) & (market_df.index < t)]
    if hist.empty:
        return Curve(np.array([]), np.array([]))
    x = hist["日前竞价空间"].to_numpy(dtype=float)
    y = hist["日前价格"].to_numpy(dtype=float)
    return _quantile_bin_curve(x, y, n_bins=30)


# -------------------- 主流程 --------------------

def _assemble_from_series(load_s: pd.Series, wind_s: pd.Series, pv_s: pd.Series) -> pd.DataFrame:
    df = pd.concat([
        _ensure_datetime_index(load_s.to_frame("load_pred")).iloc[:, 0],
        _ensure_datetime_index(wind_s.to_frame("wind_pred")).iloc[:, 0],
        _ensure_datetime_index(pv_s.to_frame("pv_pred")).iloc[:, 0],
    ], axis=1).sort_index()
    df = df.interpolate(limit_direction="both")
    df["bidspace_pred"] = df["load_pred"] - df["wind_pred"] - df["pv_pred"]
    return df


def assemble_bidspace_for_date(
    target_date: str,
    load_path: str,
    wind_path: str,
    pv_path: str,
) -> pd.DataFrame:
    """旧接口：从 dayplus parquet 读取并计算竞价空间。"""
    load_s = read_dayplus_parquet(load_path, target_date).rename("load_pred")
    wind_s = read_dayplus_parquet(wind_path, target_date).rename("wind_pred")
    pv_s   = read_dayplus_parquet(pv_path,   target_date).rename("pv_pred")
    return _assemble_from_series(load_s, wind_s, pv_s)


def baseline_predict_for_date(
    target_date: str,
    market_parquet_path: str,
    load_path: Optional[str],
    wind_path: Optional[str],
    pv_path: Optional[str],
    d_hist_days: int = 7,
    *,  # 下面为可选：传入 Series 覆盖路径（用于 "shengdiao"）
    load_series: Optional[pd.Series] = None,
    wind_series: Optional[pd.Series] = None,
    pv_series:   Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Curve, pd.Series]:
    """
    针对某个目标日期：
      1) 读取 dayplus 预测并拼成竞价空间（或直接用传入的 Series）
      2) 用 baseline 曲线（过去 d 天）进行“日前价格”预测
      3) 返回：(df_pred, curve_da, price_pred_da)
    """
    # 1) 读取市场数据（历史）
    market_df = load_market_for_history(market_parquet_path)

    # 2) 组装目标日的竞价空间
    if load_series is not None or wind_series is not None or pv_series is not None:
        if load_series is None:
            load_series = read_dayplus_parquet(load_path, target_date).rename("load_pred")
        if wind_series is None:
            wind_series = read_dayplus_parquet(wind_path, target_date).rename("wind_pred")
        if pv_series is None:
            pv_series = read_dayplus_parquet(pv_path, target_date).rename("pv_pred")
        df_pred = _assemble_from_series(load_series, wind_series, pv_series)
    else:
        df_pred = assemble_bidspace_for_date(target_date, load_path, wind_path, pv_path)

    # 3) 基线曲线（历史 d 天）
    curve_da = build_baseline_curve_da(market_df, target_date, d_hist_days=d_hist_days)

    # 4) 对目标日做价格预测
    price_pred = predict_from_curve(df_pred["bidspace_pred"].to_numpy(), curve_da)
    price_pred = pd.Series(price_pred, index=df_pred.index, name="price_pred_da")

    return df_pred, curve_da, price_pred


def save_curve_and_preds(
    out_dir: str,
    target_date: str,
    curve_da: Curve,
    price_pred_da: pd.Series,
):
    os.makedirs(out_dir, exist_ok=True)
    dstr = pd.to_datetime(target_date).strftime("%Y-%m-%d")

    # 1) 曲线保存：两列 csv（x_mid, y_mean）
    curve_path = os.path.join(out_dir, f"bidspace_curve_da_{dstr}.csv")
    curve_df = pd.DataFrame({"x_mid": curve_da.x_mid, "y_mean": curve_da.y_mean})
    curve_df.to_csv(curve_path, index=False, encoding="utf-8")

    # 2) 预测时序保存
    price_path = os.path.join(out_dir, f"price_pred_da_{dstr}.csv")
    price_pred_da.to_csv(price_path, header=True)

    return curve_path, price_path
