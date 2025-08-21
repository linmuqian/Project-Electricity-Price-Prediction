#!/usr/bin/env bash
# run_scheduler.sh —— 在当前目录运行 run_automatic_hiking_predict.py
# 用法：./run_scheduler.sh {start|stop|status|tail}

set -euo pipefail

# 进入脚本所在目录（与 Python 文件同目录）
cd "$(dirname "$0")"

# ===== 可调参数（相对路径 & 环境变量）=====
PYTHON="${PYTHON:-python}"      # 用当前环境里的 python；需要的话 export PYTHON=.../bin/python
SCRIPT="./run_daily.py"
LOG_DIR="./log"
LOG_FILE="$LOG_DIR/scheduler.log"
PID_FILE="$LOG_DIR/scheduler.pid"
LOCK_FILE="./.scheduler.lock"   # 相对路径锁文件

# 每天运行的时间/时区（传给 Python）
export RUN_TZ="${RUN_TZ:-Asia/Shanghai}"
export RUN_HOUR="${RUN_HOUR:-9}"
export RUN_MIN="${RUN_MIN:-0}"

mkdir -p "$LOG_DIR"

case "${1:-start}" in
  start)
    # 已在跑就不重复启动
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Already running (PID $(cat "$PID_FILE"))."
      exit 0
    fi
    # 启动：优先用 flock 防重；没有 flock 就直接跑
    if command -v flock >/dev/null 2>&1; then
      nohup env RUN_TZ="$RUN_TZ" RUN_HOUR="$RUN_HOUR" RUN_MIN="$RUN_MIN" \
        flock -n "$LOCK_FILE" $PYTHON "$SCRIPT" >> "$LOG_FILE" 2>&1 &
    else
      nohup env RUN_TZ="$RUN_TZ" RUN_HOUR="$RUN_HOUR" RUN_MIN="$RUN_MIN" \
        $PYTHON "$SCRIPT" >> "$LOG_FILE" 2>&1 &
    fi
    echo $! > "$PID_FILE"
    echo "Started. PID $(cat "$PID_FILE"). Logs: $LOG_FILE"
    ;;
  stop)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      kill "$(cat "$PID_FILE")" || true
      rm -f "$PID_FILE" "$LOCK_FILE"
      echo "Stopped."
    else
      echo "Not running."
      rm -f "$PID_FILE" "$LOCK_FILE" 2>/dev/null || true
    fi
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Running (PID $(cat "$PID_FILE"))."
    else
      echo "Not running."
    fi
    ;;
  tail)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|tail}"
    exit 2
    ;;
esac
