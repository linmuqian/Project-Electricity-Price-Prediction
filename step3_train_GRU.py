# train_GRU.py
# ------------------------------------------------------------
# 功能：基于“今天”的数据训练/验证，并对“明天”做单日预测（不滚动）
# 特点：
#   - 可通过 --dataset_ver 选择 utils.hiking_dataset_V2 / V2_4（两者均已支持推理）
#   - 测试集走 inference_allow_unlabeled=True：明天即使无真标签也能预测
#   - 支持指定时区/今天日期、训练/验证窗口长度、结果保存
# 依赖：
#   - scripts/train_single_GRU.py 的 main()
# ------------------------------------------------------------
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import pytz

# 运行路径建议在项目根目录执行：python scripts/train_GRU_today.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_single_GRU import main as train_single_GRU  # 单次训练+预测的入口


def to_str(d: datetime) -> str:
    """将 date/datetime 转为 YYYY-MM-DD 字符串。"""
    return d.strftime("%Y-%m-%d")


def parse_args():
    """参数解析"""
    p = argparse.ArgumentParser(description="基于今天→明天的单日预测（不滚动）")

    # --- 时间相关 ---
    p.add_argument("--today", type=str, default=None, help="今天的日期(YYYY-MM-DD)。默认按 --tz 取系统当前日期。")
    p.add_argument("--tz", type=str, default="Asia/Shanghai", help="时区(默认 Asia/Shanghai)，如 'UTC'、'America/Los_Angeles'。")

    # --- 窗口长度：训练、验证 ---
    p.add_argument("--train_days", type=int, default=360, help="训练窗口天数(默认360天)，结束于验证集前一天。")
    p.add_argument("--val_days", type=int, default=14, help="验证窗口天数(默认14天)，结束于 today。")

    # --- 边界保护 ---
    p.add_argument("--min_train_begin", type=str, default=None, help="可选：强制训练集起始不早于该日期(YYYY-MM-DD)。")

    # --- 数据集版本 ---
    p.add_argument("--dataset_ver", type=str, default="V2", choices=["V2", "V2_4"],
                   help="选择数据集版本：utils.hiking_dataset_V2 或 utils.hiking_dataset_V2_4。")

    # --- 输出 ---
    p.add_argument("--save_csv", action="store_false", help="是否把预测结果另存为 CSV（默认仅打印）。")
    p.add_argument("--out_dir", type=str, default="save/today_pred", help="保存目录（当 --save_csv 打开时使用）。")

    return p.parse_args()


def main():
    args = parse_args()

    # 1) 设定 today
    if args.today is None:
        tz = pytz.timezone(args.tz)
        today = datetime.now(tz).date()
    else:
        today = datetime.strptime(args.today, "%Y-%m-%d").date()

    # 2) 目标测试日为“明天”（单日预测）
    test_day = today + timedelta(days=1)

    # 3) 验证集：最近 val_days 天，结束于 today
    val_end = today
    val_begin = today - timedelta(days=args.val_days - 1)

    # 4) 训练集：位于验证集前面，跨度 train_days，结束于 val_begin 前一天
    train_end = val_begin - timedelta(days=1)
    train_begin = train_end - timedelta(days=args.train_days - 1)

    # 5) 可选：训练集起始边界保护（避免越界到数据缺失区间）
    if args.min_train_begin is not None:
        guard = datetime.strptime(args.min_train_begin, "%Y-%m-%d").date()
        if train_begin < guard:
            train_begin = guard

    # 6) 组装字符串日期（下游 main() 需要 YYYY-MM-DD）
    train_begin_s = to_str(datetime.combine(train_begin, datetime.min.time()))
    train_end_s   = to_str(datetime.combine(train_end,   datetime.min.time()))
    val_begin_s   = to_str(datetime.combine(val_begin,   datetime.min.time()))
    val_end_s     = to_str(datetime.combine(val_end,     datetime.min.time()))
    test_begin_s  = to_str(datetime.combine(test_day,    datetime.min.time()))
    test_end_s    = test_begin_s  # 单日预测：起止相同

    # 7) 打印配置，便于核对
    print("[配置]")
    print(f"  today        = {today}  (tz={args.tz})")
    print(f"  dataset_ver  = {args.dataset_ver}")
    print(f"  train        = {train_begin_s}  →  {train_end_s}    (~{args.train_days} 天)")
    print(f"  valid        = {val_begin_s}    →  {val_end_s}      (~{args.val_days} 天)")
    print(f"  test(single) = {test_begin_s}")
    print("-" * 60)

    # 8) 调用单次训练-预测入口
    #    注意：train_single_GRU() 内部会在测试集启用 inference_allow_unlabeled=True，
    #          因此明天即使没有真标签也能产样本并输出 Pred_Label / Pred_Probability。
    try:
        predict_df = train_single_GRU(
            train_begin=train_begin_s, train_end=train_end_s,
            valid_begin=val_begin_s,   valid_end=val_end_s,
            test_begin=test_begin_s,   test_end=test_end_s,
            dataset_ver=args.dataset_ver,  
        )
    except Exception as e:
        print(f"[ERROR] 预测失败：{e}")
        print("可能原因：明天的日前特征缺失/不完整；或历史不足以构造 sequence_length。")
        return

    # 9) 打印结果（含概率）。若无真标签列，则只显示预测与概率。
    print("\n[明日预测结果]")
    cols = [c for c in ["Date", "True_Label", "Pred_Label", "Pred_Probability"] if c in predict_df.columns]
    try:
        print(predict_df[cols].to_string(index=False))
    except Exception:
        print(predict_df.head())

    # 10) 可选：保存结果为 CSV（文件名包含数据集版本与日期）
    if args.save_csv:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, f"today_pred_hiking_{args.dataset_ver}_{test_begin_s}.csv")
        predict_df.to_csv(out_path, index=False)
        print(f"\n已保存到: {out_path}")


if __name__ == "__main__":
    main()
