# run_automatic_hiking_predict.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ========= 可配项 =========
RUN_TZ    = os.environ.get("RUN_TZ", "Asia/Shanghai")  # 业务时区
RUN_HOUR  = int(os.environ.get("RUN_HOUR", 9))         # 每天几点跑
RUN_MIN   = int(os.environ.get("RUN_MIN", 0))          # 每天几分跑

REPO_ROOT = Path(__file__).resolve().parent            # 项目根目录
SCRIPTS = [                                            # 依次运行的脚本
    REPO_ROOT / "step1_update_market_data.py",
    REPO_ROOT / "step2_dataset_generate.py",
    REPO_ROOT / "step3_train_GRU.py",
    REPO_ROOT / "step4_price_prediction.py",  
]
LOG_DIR   = REPO_ROOT / "log_daily"                    # 日志目录
PY        = sys.executable                             # 使用当前解释器

# 如三步都需要同一个日期参数（例：明天的日期），可开启这个函数并把 args 传入 run_step
# 但由于step1脚本自动读取当日日期，step2和step3脚本读取之前step的输出结果，因此不需要传参
def common_args_for_today() -> list[str]:
    # 示例：把“明天”作为业务日传给子脚本
    # tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    # return [f"--date={tomorrow}"]
    return []

# ========= 实用函数 =========
def run_step(script_path: Path, args: list[str] | None = None):
    args = args or []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{script_path.stem}_{ts}.log"

    cmd = [PY, str(script_path), *args]
    print(f"\n[{ts}] Running: {' '.join(cmd)}\nLog: {log_file}")

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\nCWD: {REPO_ROOT}\n\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in p.stdout:
            print(line, end="")
            lf.write(line)
        p.wait()
        rc = p.returncode

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

def seconds_until_next_run(now: datetime) -> float:
    target = now.replace(hour=RUN_HOUR, minute=RUN_MIN, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()

def main():
    # 设置时区（Linux 有效；Windows 忽略即可）
    os.environ["TZ"] = RUN_TZ
    try:
        import time as _t; _t.tzset()
    except Exception:
        pass

    print(f"Scheduler started. Daily at {RUN_HOUR:02d}:{RUN_MIN:02d} ({RUN_TZ}).")

    shared_args = common_args_for_today()  # 若不需要统一参数，函数里直接返回 []

    while True:
        # 休眠到下一个触发点
        sleep_s = seconds_until_next_run(datetime.now())
        while sleep_s > 0:
            time.sleep(min(300, sleep_s))  # 每5分钟醒一次，便于手动停止
            sleep_s -= 300

        # 到点后跑三步
        print(f"\n=== {datetime.now():%F %T} start daily run ===")
        try:
            for script in SCRIPTS:
                run_step(script, shared_args)
            print(f"=== {datetime.now():%F %T} done (OK) ===\n")
        except Exception as e:
            print(f"=== {datetime.now():%F %T} failed: {e} ===\n")

        # 跑完后继续等待下一天
        time.sleep(1)

if __name__ == "__main__":
    main()
