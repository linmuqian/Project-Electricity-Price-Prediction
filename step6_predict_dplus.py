
# predict_dplus_baseline.py
# ------------------------------------------------------------
# 功能：用 baseline（过去 d 天）“日前竞价空间→价格”静态曲线
#       预测 D+2..D+5 的日前价格（D+1这里不处理）。
# 依赖：utils/dplus_utils.py
# ------------------------------------------------------------
# predict_dplus_baseline.py  

from __future__ import annotations
import os
from datetime import datetime, timedelta
import pytz

# --- 日志目录兜底（部分依赖库在导入时需要）---
_logdir = os.path.expanduser("~/.pytsingrocbytelinker_logs")
os.makedirs(_logdir, exist_ok=True)
for k in ("LOG_ROOT", "PYTSINGROCBYTELINKER_LOG_ROOT", "TSINGROC_LOG_ROOT", "BYTE_LOG_ROOT"):
    os.environ.setdefault(k, _logdir)

# DB 读取（用于在线更新 parquet；若不需要可关掉 enable_update）
try:
    from pytsingrocbytelinker import PostgreLinker, DbType, DBInfo
except Exception:
    PostgreLinker = None
    DbType = None
    DBInfo = None

from utils.dplus_utils import (
    baseline_predict_for_date,
    save_curve_and_preds,
    read_shengdiao_dn_series,
)


CONFIG = {
    # AUTO指的是北京时间的今天
    # 也可以改成2025-08-10等
    "run_date": "AUTO",
    # 历史市场数据（用于拟合静态曲线）
    "market": "data/processed/shanxi_new.parquet",
    # 省调 Dn 数据（shengdiao 模式用）
    "market_dn_path": "data/market_Dn_data.parquet",
    "pred_root": "data/processed",
    "out_root": "save/dplus_pred",
    "d": 7,

    # 在线更新 dayplus parquet，不开则用本地现有文件
    "enable_update": False, # 是否自动更新风光负荷data/processed/wind/dayplus_{2..5}_{label}.parquet等
    "dbtype": "test",
    "region": "cn-shanxi",

    "wind_sources": {
        "gfs": "清鹏-lgbm-V1.0.0-zhw-gfs",
        "ifs": "清鹏-lgbm-V1.0.0-zhw-ecmwfifs025",
    },
    "pv_sources": {
        "gfs": "清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源gfsglobal",
        "ifs": "清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源ecmwfifs025",
    },
    "load_sources": {
        "gfs": "清鹏-lasso_lgbm_mlp-gfs-V1.0.0-xzq",
        "ifs": "清鹏-lasso_lgbm_mlp-ec-V1.0.0-xzq",
    },

    # 现在 label 允许：'gfs' | 'ifs' | 'shengdiao'
    # shengdiao 指的是从 market_Dn_data.parquet 读取 D plus 当天的风光负荷数据
    "load_label": "gfs",
    "wind_label": "shengdiao",
    "pv_label":   "gfs",
}


# -------------------- DB 读取封装（用于更新 parquet） --------------------
def _connect(dbtype: str):
    linker = PostgreLinker()
    if dbtype == "test":
        linker.connect(DbType.test)
    elif dbtype == "prod":
        linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    return linker


def _build_pred_path(root: str, kind: str, dplus: int, label: str):
    return os.path.join(root, kind, f"dayplus_{dplus}_{label}.parquet")


def _update_one(kind: str, h: int, source_name: str, cfg):
    # kind in {'wind','pv','load'}
    if PostgreLinker is None:
        raise RuntimeError("未安装/无法导入 pytsingrocbytelinker，无法在线更新。")
    linker = _connect(cfg["dbtype"])
    db_info = DBInfo()
    if kind in ("wind", "pv"):
        db_info.read_from_data(
            table_name="power_pred_province",
            key=f"pred_{kind}",
            tags={"source": source_name, "day_plus": h},
            daypoints=96,
            version="v1.0.0",
            column_name="pred",
            start_time="2025-01-01 00:00:00",
            end_time=(run_date + timedelta(days=5)).strftime("%Y-%m-%d 23:45:00"),
            region=cfg["region"],
            value_tag="value",
        )
    else:  # load
        db_info.read_from_data(
            table_name="load_pred",
            key="pred_load",
            tags={"source": source_name, "dayplus": h},
            daypoints=96,
            version="v1.0.0",
            column_name="pred",
            start_time="2025-01-01 00:00:00",
            end_time=(run_date + timedelta(days=5)).strftime("%Y-%m-%d 23:45:00"),
            region=cfg["region"],
            value_tag="value",
        )
    df = linker.query(db_info)
    linker.disconnect()
    df["time"] = df["time"].dt.tz_localize(None)
    outp = _build_pred_path(cfg["pred_root"], kind, h, label=label_for_kind[kind])
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    df.to_parquet(outp, engine="fastparquet")
    print(f"[更新] {kind} D+{h} → {outp}")


def update_dayplus_parquets(run_date, cfg):
    horizons = [2, 3, 4, 5]
    # 跳过 shengdiao（它从 market_Dn_data 拿，不需要在线更新）
    for kind in ("wind", "pv", "load"):
        lab = label_for_kind[kind]
        if lab == "shengdiao":
            print(f"[INFO] 跳过在线更新 {kind}（使用 shengdiao 数据）。")
            continue
        source_map = cfg[f"{kind}_sources"]
        if lab not in source_map:
            print(f"[WARN] 跳过 {kind}：label={lab} 在 sources 中未配置。")
            continue
        for h in horizons:
            try:
                _update_one(kind, h, source_map[lab], cfg)
            except Exception as e:
                print(f"[WARN] {kind} D+{h} 未更新：{e}")


# -------------------- 预测块 --------------------
def _resolve_market_path(p):
    if os.path.exists(p): 
        return p
    alt = os.path.join("data", "processed", "shanxi_new.parquet")
    if p != alt and os.path.exists(alt):
        print(f"[INFO] 未找到 {p} ，改用 {alt}")
        return alt
    raise FileNotFoundError(f"找不到行情 parquet：{p}（也不存在 {alt}）")


def baseline_predict_block(run_date, cfg):
    print(f"[INFO] 运行参考日 = {run_date}；将预测 D+2..D+5：",
          [run_date + timedelta(days=h) for h in (2,3,4,5)])

    horizons = [2, 3, 4, 5]

    for h in horizons:
        target_date = (run_date + timedelta(days=h)).strftime("%Y-%m-%d")
        print(f"\n[INFO] 开始预测 D+{h} → 目标日 {target_date}")

        # 1) 按 label 选择数据来源
        load_series = wind_series = pv_series = None
        # shengdiao → 从 market_Dn_data 读取 "D 当天 *_d{h}"
        if label_for_kind["load"] == "shengdiao":
            load_series = read_shengdiao_dn_series("load", base_date=run_date, horizon=h, market_dn_parquet_path=cfg["market_dn_path"])
        if label_for_kind["wind"] == "shengdiao":
            wind_series = read_shengdiao_dn_series("wind", base_date=run_date, horizon=h, market_dn_parquet_path=cfg["market_dn_path"])
        if label_for_kind["pv"] == "shengdiao":
            pv_series   = read_shengdiao_dn_series("pv",   base_date=run_date, horizon=h, market_dn_parquet_path=cfg["market_dn_path"])

        # 非 shengdiao → dayplus parquet 路径
        load_path = None if load_series is not None else _build_pred_path(cfg["pred_root"], "load", h, label_for_kind["load"])
        wind_path = None if wind_series is not None else _build_pred_path(cfg["pred_root"], "wind", h, label_for_kind["wind"])
        pv_path   = None if pv_series   is not None else _build_pred_path(cfg["pred_root"], "pv",   h, label_for_kind["pv"])

        try:
            df_pred, curve_da, price_pred_da = baseline_predict_for_date(
                target_date=target_date,
                market_parquet_path=cfg["market"],
                load_path=load_path,
                wind_path=wind_path,
                pv_path=pv_path,
                d_hist_days=int(cfg["d"]),
                load_series=load_series,
                wind_series=wind_series,
                pv_series=pv_series,
            )
        except Exception as e:
            print(f"[WARN] D+{h} 失败：{e}")
            continue

        out_dir = os.path.join(cfg["out_root"], f"D+{h}")
        curve_path, price_path = save_curve_and_preds(out_dir, target_date, curve_da, price_pred_da)
        try:
            bidspace_path = os.path.join(out_dir, f"bidspace_pred_{target_date}.csv")
            df_pred.to_csv(bidspace_path)
        except Exception:
            bidspace_path = "(未保存)"
        print(f"[OK] D+{h} 完成：\n      曲线 → {curve_path}\n      价格 → {price_path}\n      竞价空间 → {bidspace_path}")


# -------------------- main --------------------
def main():
    global run_date, label_for_kind
    cfg = CONFIG.copy()
    # run_date
    if str(cfg["run_date"]).upper() == "AUTO":
        beijing = pytz.timezone("Asia/Shanghai")
        run_date = datetime.now().astimezone(beijing).date()
    else:
        run_date = datetime.fromisoformat(str(cfg["run_date"])).date()

    # 将 label 汇总便于使用
    label_for_kind = {
        "load": cfg["load_label"],
        "wind": cfg["wind_label"],
        "pv":   cfg["pv_label"],
    }

    # 校验/兜底 market 路径
    cfg["market"] = _resolve_market_path(cfg["market"])

    # Step 1: 在线更新 parquet（如果开启且 label 不是 shengdiao）
    if cfg.get("enable_update", False):
        try:
            update_dayplus_parquets(run_date, cfg)
        except Exception as e:
            print(f"[WARN] 更新步骤失败或不可用，跳过在线更新：{e}")

    # Step 2: baseline 预测
    baseline_predict_block(run_date, cfg)


if __name__ == "__main__":
    main()
