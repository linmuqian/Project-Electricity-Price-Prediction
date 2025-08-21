import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import calendar


def find_decade_dates(
    year: int,
    month: int,
    day: int = 1,
) -> Tuple[int, str, str]:
    """
    根据年月日和旬的定义，返回旬的开始和结束日期

    Args:
        year (int): 年份
        month (int): 月份
        decade (Decade, optional): 旬. Defaults to Decade.EARLY.
        day (int, optional): 日期. Defaults to 1.
        format (DateFormat, optional): 格式. Defaults to DateFormat.DECADE.

    Returns:
        Tuple[int, str, str]: 旬的天数、开始日期、结束日期
    """

    # 确定月份的天数
    num_days = calendar.monthrange(year, month)[1]
    if day <= 10:
        start_day = 1
        end_day = 10
    elif day <= 20:
        start_day = 11
        end_day = 20
    else:
        start_day = 21
        end_day = num_days
    # 生成日期列表
    return (
        end_day - start_day + 1,
        f"{year}-{month:02d}-{start_day:02d}",
        f"{year}-{month:02d}-{end_day:02d}",
    )
