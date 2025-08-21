# 自动运行全流程的pipeline脚本

from pathlib import Path
import subprocess
import sys
from datetime import datetime
import os

def run_step(script_path: Path, args=None, cwd=None, log_dir: Path = None):
    args = args or []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_path.stem}_{ts}.log"

    cmd = [sys.executable, str(script_path), *args]
    print(f"\n=== Running: {' '.join(cmd)} ===")
    print(f"Log: {log_file}")

    # 实时读取 stdout（包含 stderr），边打印边写日志
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\nCWD: {cwd or os.getcwd()}\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            print(line, end="")   # 终端实时可见
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print(f"=== Done: {script_path.name} ===")
    return log_file

def main():
    repo_root = Path(__file__).resolve().parent           # 放在项目根目录
    dir = repo_root 
    log_dir = repo_root / "log" / rf"pipeline_{datetime.now().strftime('%Y%m%d')}"

    steps = [
        (dir / "step1_update_market_data.py", []),
        (dir / "step2_dataset_generate.py", []),
        (dir / "step3_train_GRU.py", []),
        (dir / "step4_price_prediction.py", []),  
    ]

    # 如果三个脚本需要同一个参数（例如 --date），可在这里统一加到 args
    # e.g. common_args = ["--date=2025-08-18"]
    common_args = []

    for script, args in steps:
        run_step(script, args + common_args, cwd=repo_root, log_dir=log_dir)

    print("\nAll steps completed successfully ✅")

if __name__ == "__main__":
    main()
