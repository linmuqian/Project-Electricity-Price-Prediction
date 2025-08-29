'''
1、更新QR_pred.ipynb中生成的【风】【光】【负荷】的D+预测数据
2、更新【省调】update_Dn_market_data.py中生成的market_Dn_data.parquet
'''
### QR_pred.ipynb

import os
os.environ.setdefault("LOG_ROOT", "log")

from pytsingrocbytelinker import PostgreLinker, DbType, WriteData, DBInfo


import pandas as pd
from pytsingrocbytelinker import PostgreLinker, DbType, WriteData, DBInfo

def read_pred_pv_province(dayplus, dbtype, source, date_begin, date_end) -> None:
    byte_linker = PostgreLinker()
    if dbtype == 'test':
        byte_linker.connect(DbType.test)
    elif dbtype == 'prod':
        byte_linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    db_info = DBInfo()
    db_info.read_from_data(
        table_name="power_pred_province",
        key="pred_pv",
        tags={ "source": source, "day_plus": dayplus},
        daypoints=96,
        version="v1.0.0",
        column_name="pred",
        start_time=date_begin,
        end_time=date_end,
        region="cn-shanxi",
        value_tag="value",
    )
    data = byte_linker.query(db_info)
    byte_linker.disconnect()
    data['time'] = data['time'].dt.tz_localize(None)
    return data


def read_pred_wind_province(dayplus, dbtype, source, date_begin, date_end) -> None:
    byte_linker = PostgreLinker()
    if dbtype == 'test':
        byte_linker.connect(DbType.test)
    elif dbtype == 'prod':
        byte_linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    db_info = DBInfo()
    db_info.read_from_data(
        table_name="power_pred_province",
        key="pred_wind",
        tags={ "source": source, "day_plus": dayplus},
        daypoints=96,
        version="v1.0.0",
        column_name="pred",
        start_time= date_begin,   #  "2025-01-01 00:00:00"
        end_time=date_end,  #"2025-05-10 23:45:00",
        region="cn-shanxi",
        value_tag="value",
    )
    data = byte_linker.query(db_info)
    byte_linker.disconnect()
    data['time'] = data['time'].dt.tz_localize(None)
    return data

def read_pred_load_province(dayplus, dbtype, source, date_begin, date_end) -> None:
    byte_linker = PostgreLinker()
    if dbtype == 'test':
        byte_linker.connect(DbType.test)
    elif dbtype == 'prod':
        byte_linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    db_info = DBInfo()
    db_info.read_from_data(
        table_name="load_pred",
        key="pred_load",
        tags={ "source": source, "dayplus": dayplus},
        daypoints=96,
        version="v1.0.0",
        column_name="pred",
        start_time= date_begin,   #  "2025-01-01 00:00:00"
        end_time=date_end,  #"2025-05-10 23:45:00",
        region="cn-shanxi",
        value_tag="value",
    )
    data = byte_linker.query(db_info)
    byte_linker.disconnect()
    data['time'] = data['time'].dt.tz_localize(None)
    return data

## 风
dbtype = "test"
date_begin = "2025-01-01 00:00:00"

from datetime import datetime, timedelta
import pytz

beijing = pytz.timezone("Asia/Shanghai")
today_bj = datetime.now(beijing).date()
date_end = (today_bj + timedelta(days=10)).strftime("%Y-%m-%d") + " 23:45:00"


wind_dict = {
    "gfs":"清鹏-lgbm-V1.0.0-zhw-gfs",
    "ifs":"清鹏-lgbm-V1.0.0-zhw-ecmwfifs025",
    #"aifs":"清鹏-lgbm-V1.0.0-zhw-ecmwfaifs025"
}

for label,tag in wind_dict.items():
    for dayplus in range(1,6):
        try:
            wind_pred = read_pred_wind_province(dayplus, dbtype, tag, date_begin, date_end)
            wind_pred.to_parquet(f"data/processed/wind/dayplus_{dayplus}_{label}.parquet", engine='fastparquet')
        except Exception as e:
            print(f"label: {label}, tag: {tag}, dayplus: {dayplus} not found, error: {e}")
            continue
        
## 光

dbtype = "test"
date_begin = "2025-01-01 00:00:00"

from datetime import datetime, timedelta
import pytz

beijing = pytz.timezone("Asia/Shanghai")
today_bj = datetime.now(beijing).date()
date_end = (today_bj + timedelta(days=10)).strftime("%Y-%m-%d") + " 23:45:00"

pv_dict = {
    "gfs":"清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源gfsglobal",
    "ifs":"清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源ecmwfifs025",
    #"aifs":"清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源ecmwfaifs025"
}

for label,tag in pv_dict.items():
    for dayplus in range(1,6):
        try:
            pv_pred = read_pred_pv_province(dayplus, dbtype, tag, date_begin, date_end)
            pv_pred.to_parquet(f"data/processed/pv/dayplus_{dayplus}_{label}.parquet", engine='fastparquet')
        except Exception as e:
            print(f"label: {label}, tag: {tag}, dayplus: {dayplus} not found, error: {e}")
            continue


## 负荷

dbtype = "test"
date_begin = "2025-01-01 00:00:00"

from datetime import datetime, timedelta
import pytz

beijing = pytz.timezone("Asia/Shanghai")
today_bj = datetime.now(beijing).date()
date_end = (today_bj + timedelta(days=10)).strftime("%Y-%m-%d") + " 23:45:00"


load_dict = {
    "gfs":"清鹏-lasso_lgbm_mlp-gfs-V1.0.0-xzq",
    "ifs":"清鹏-lasso_lgbm_mlp-ec-V1.0.0-xzq"
}

for label,tag in load_dict.items():
    for dayplus in range(1,6):
        try:
            load_pred = read_pred_load_province(dayplus, dbtype, tag, date_begin, date_end)
            load_pred.to_parquet(f"data/processed/load/dayplus_{dayplus}_{label}.parquet", engine='fastparquet')
        except Exception as e:
            print(f"label: {label}, tag: {tag}, dayplus: {dayplus} not found, error: {e}")
            continue
        
        
dbtype = "test"
date_begin = "2025-01-01 00:00:00"

from datetime import datetime, timedelta
import pytz

beijing = pytz.timezone("Asia/Shanghai")
today_bj = datetime.now(beijing).date()
date_end = (today_bj + timedelta(days=10)).strftime("%Y-%m-%d") + " 23:45:00"


for dayplus in range(4,15):
    #pv_pred = read_pred_pv_province(dayplus, dbtype, "清鹏-lgbm-V3.0.0-zxh-预测不限电发电量-单气象源ecmwfifs025", date_begin, date_end)
    #wind_pred = read_pred_wind_province(dayplus, dbtype, '清鹏-lgbm-V1.0.0-zhw-gfs', date_begin, date_end)
    load_pred = read_pred_load_province(dayplus, dbtype, '清鹏', date_begin, date_end)
    #pv_pred.to_parquet(f"../data/processed/pv/dayplus_{dayplus}_V3.parquet", engine='fastparquet')
    #wind_pred.to_parquet(f"../data/processed/wind/dayplus_{dayplus}_V1.parquet", engine='fastparquet')
    load_pred.to_parquet(f"data/processed/load/dayplus_{dayplus}_V0.parquet", engine='fastparquet')


### update_Dn_market_data.py

from datetime import datetime, timedelta, time
import time as TIME
import pytz
import pandas as pd
import warnings
import sys
import json
import os


# from real_capacity.capacity_new import load_online_capacity
from utils.download_multiday_data_simple import read_intraday_data, read_dayahead_data, get_d0_and_dplus_data
# from utils.upload_data import write_real_data, write_pred_data, write_price_real_intra, write_real_newpower, write_pred_newpower

# 忽略指定的警告
warnings.simplefilter("ignore", UserWarning)

# 获取当前日期和时间
beijing_tz = pytz.timezone('Asia/Shanghai')
current_datetime = datetime.now().astimezone(beijing_tz)
print("当天日期：", current_datetime)

def process_combined_data(combined_data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    处理合并后的数据，主要是对每天一值的列进行处理

    Args:
        combined_data (pd.DataFrame): 合并后的数据
        columns (list): 需要处理的列名列表

    Returns:
        pd.DataFrame: 处理后的数据
    """
    # 处理正备用和新能源负荷备用，将每天一个值的列填充到全天
    # 这里要注意，除了实时特征、日前在线机组容量和日前负荷率，其他的多日数据都是由dayplus的了

    # 处理每日一值的特征，仅对"正备用-日前(MW)"、"新能源负荷备用-日前(MW)"和"日前在线机组容量(MW)"填充全天
    # 只对"正备用-日前(MW)_d1/2/3"和"新能源负荷备用-日前(MW)_d1/2/3"按天填充，'日前在线机组容量(MW)'只处理一次
    for dayplus in range(1, 4):
        for column in [f"正备用-日前(MW)_d{dayplus}", f"新能源负荷备用-日前(MW)_d{dayplus}"]:
            if column in combined_data.columns:
                daily_values = combined_data[column].resample('D').first()
                for day, value in daily_values.items():
                    if pd.notna(value):
                        day_start = day.replace(hour=0, minute=0, second=0)
                        day_end = day.replace(hour=23, minute=59, second=59)
                        combined_data.loc[day_start:day_end, column] = value

    # 只处理一次"日前在线机组容量(MW)"，并且只对其每日一值填充全天
    if '日前在线机组容量(MW)' in combined_data.columns:
        daily_values = combined_data['日前在线机组容量(MW)'].resample('D').first()
        for day, value in daily_values.items():
            if pd.notna(value):
                day_start = day.replace(hour=0, minute=0, second=0)
                day_end = day.replace(hour=23, minute=59, second=59)
                combined_data.loc[day_start:day_end, '日前在线机组容量(MW)'] = value

    # 联络线计划：数据库中读取到的是负值，代表山西向外输电
    # 由于数据库目前未补充至20230101，之前的数据使用的是融一的为正值，所以暂时取反
    # 后续会将负号去掉，然后修改竞价空间计算方法
    for dayplus in range(1, 4):
        combined_data[f'联络线计划-日前(MW)_d{dayplus}'] = -combined_data[f'联络线计划-日前(MW)_d{dayplus}']
    if '联络线计划-日内(MW)' in combined_data.columns:
        combined_data['联络线计划-日内(MW)'] = -combined_data['联络线计划-日内(MW)']

    # 计算竞价空间（主表）
    if all(col in combined_data.columns for col in ['省调负荷-日内(MW)', '联络线计划-日内(MW)', '实时光伏(MW)', '实时风电(MW)']):
        combined_data['竞价空间-日内(MW)'] = (
            combined_data['省调负荷-日内(MW)'] +
            combined_data['联络线计划-日内(MW)'] -
            combined_data['实时光伏(MW)'] -
            combined_data['实时风电(MW)'] 
            #-combined_data['非市场化机组出力-日内(MW)']
        )

    # 计算竞价空间的dayplus特征
    # 例如：'竞价空间-日前(MW)_d1', '竞价空间-日前(MW)_d2', '竞价空间-日前(MW)_d3', ...
    for i in range(1, 4):  # 假设最多到d3
        suffix = f"_d{i}"
        load_col = f"省调负荷-日前(MW){suffix}"
        tie_col = f"联络线计划-日前(MW){suffix}"
        pv_col = f"日前光伏(MW){suffix}"
        wind_col = f"日前风电(MW){suffix}"
        target_col = f"竞价空间-日前(MW){suffix}"
        if all(col in combined_data.columns for col in [load_col, tie_col, pv_col, wind_col]):
            combined_data[target_col] = (
                combined_data[load_col] +
                combined_data[tie_col] -
                combined_data[pv_col] -
                combined_data[wind_col]
            )

    # 计算负荷率
    # 日前负荷率(%) = d+1的竞价空间-日前(MW) / d+1的日前在线机组容量(MW) * 100
    # 仅当在线机组容量不为nan且不为-999时才计算
    # 注意：只对d+1的负荷率进行计算

    mask = (combined_data['日前在线机组容量(MW)'].notna()) & (combined_data['日前在线机组容量(MW)'] != -999)
    combined_data['日前负荷率(%)'] = None  # 先置空
    combined_data.loc[mask, '日前负荷率(%)'] = (
        combined_data.loc[mask, '竞价空间-日前(MW)_d1'] / combined_data.loc[mask, '日前在线机组容量(MW)'] * 100
    )
    # 实时负荷率
    mask = (combined_data['实时在线机组容量(MW)'].notna()) & (combined_data['实时在线机组容量(MW)'] != -999)
    combined_data['实时负荷率(%)'] = None
    combined_data.loc[mask, '实时负荷率(%)'] = (
        combined_data.loc[mask, '竞价空间-日内(MW)'] / combined_data.loc[mask, '实时在线机组容量(MW)'] * 100
    )

    # 近期日前在线机组容量波动较大，以下操作暂不启用
    # combined_data = fill_capacity(combined_data, column=['日前在线机组容量(MW)'])
    # combined_data = fill_load_rate(combined_data, column=['日前负荷率(%)'])

    combined_data.index = combined_data.index.tz_localize(None)
    return combined_data.reindex(columns=columns)

def load_all_data(start_date: str = "2025-04-16 00:00:00", current_datetime: datetime = None, columns: list = None) -> pd.DataFrame:
    """
    加载所有数据的主函数

    Args:
        start_date (str): 开始日期，格式为 "YYYY-MM-DD HH:MM:SS"
        current_datetime (datetime): 当前日期时间
        columns (list): 需要返回的列名列表

    Returns:
        pd.DataFrame: 加载并处理后的数据
    """
    # 读取日内和多日数据
    d0_and_dplus_data = get_d0_and_dplus_data(start_date, current_datetime )
    from utils.download_data import read_pred_load
    # qp预测负荷
    # data_list = []
    # for dayplus in range(1, 6):
    #     data = read_pred_load(start_date=start_date, end_date=None, dayplus=dayplus, source='清鹏', dbtype='test')
    #     data_list.append(data)
    #     print(f"D{dayplus} 数据形状: {data.shape}")

    # # 直接使用concat合并，按列合并
    # merged_df = pd.concat(data_list, axis=1)
    # merged_df.index=merged_df.index.tz_localize(None)
    # 读取实时在线机组容量数据
    end_date = (current_datetime.replace(tzinfo=None) - timedelta(days=1)).replace(
    hour=23, minute=45, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")
    # 直接使用日内和多日数据，不再合并real_capacity
    combined_data = d0_and_dplus_data

    if columns is not None:
        final_data = process_combined_data(combined_data, columns)
        final_data = final_data.reindex(columns=columns)
        return final_data
    return combined_data

def check_and_upload_data(new_data: pd.DataFrame, existing_data: pd.DataFrame, start_date: str, current_datetime: datetime, columns: list, data_dir: str, write_flag: bool = True) -> dict:
    import numpy as np
    # new_data.loc['2025-07-01',['非市场化机组出力-日前(MW)_d3', '正备用-日前(MW)_d3', '新能源负荷备用-日前(MW)_d3']] = np.nan
    # 检查数据是否有异常
    intraday_columns = ['非市场化机组出力-日内(MW)', '竞价空间-日内(MW)', '实时价格', '联络线计划-日内(MW)', '频率实际值(MW)', '实际上旋备用(MW)', '实际下旋备用(MW)', 
                        '省调负荷-日内(MW)', '新能源负荷-日内(MW)', '实时风电(MW)', '实时光伏(MW)', '水电出力值-日内(MW)', '实时在线机组容量(MW)', '实时负荷率(%)']
    d_columns_dict = {
        0: [
        'price',
        '日前在线机组容量(MW)'
        ],
        1: [
            "非市场化机组出力-日前(MW)_d1",
            "联络线计划-日前(MW)_d1",
            "竞价空间-日前(MW)_d1",
            "省调负荷-日前(MW)_d1",
            "新能源负荷-日前(MW)_d1",
            "正备用-日前(MW)_d1",
            "新能源负荷备用-日前(MW)_d1",
            "日前光伏(MW)_d1",
            "日前风电(MW)_d1"
        ],
        2: [
            "非市场化机组出力-日前(MW)_d2",
            "联络线计划-日前(MW)_d2",
            "竞价空间-日前(MW)_d2",
            "省调负荷-日前(MW)_d2",
            "新能源负荷-日前(MW)_d2",
            "正备用-日前(MW)_d2",
            "新能源负荷备用-日前(MW)_d2",
            "日前光伏(MW)_d2",
            "日前风电(MW)_d2"
        ],
        3: [
            "非市场化机组出力-日前(MW)_d3",
            "联络线计划-日前(MW)_d3",
            "竞价空间-日前(MW)_d3",
            "省调负荷-日前(MW)_d3",
            "新能源负荷-日前(MW)_d3",
            "正备用-日前(MW)_d3",
            "新能源负荷备用-日前(MW)_d3",
            "日前光伏(MW)_d3",
            "日前风电(MW)_d3"
        ],
        4: [
            "日前光伏(MW)_d4",
            "日前风电(MW)_d4"
        ],
        5: [
            "日前光伏(MW)_d5",
            "日前风电(MW)_d5"
        ]
    }
    multi_dayplus_check={1:True,2:True,3:True,4:True,5:True}
    today_columns = ['price', '日前在线机组容量(MW)', '日前负荷率(%)']
    start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    
    # 数据检查
    # 检查日内特征是否有昨日的数据
    yesterday = (current_datetime - timedelta(days=1)).date()
    yesterday_start = datetime.combine(yesterday, time(hour=0, minute=0, second=0))
    yesterday_end = datetime.combine(yesterday, time(hour=23, minute=45, second=0))
    intraday_data_check = new_data.loc[yesterday_start:yesterday_end, intraday_columns]
    if intraday_data_check.empty:
        intraday_data_complete = False
        multi_dayplus_check[1]=False
    else:
        intraday_data_complete = not intraday_data_check.isna().any().any()

    # 检查今日特征是否有今日的数据
    today = current_datetime.date()
    today_start = datetime.combine(today, time(hour=0, minute=0, second=0))
    today_end = datetime.combine(today, time(hour=23, minute=45, second=0))
    today_data_check = new_data.loc[today_start:today_end, today_columns]
    if today_data_check.empty:
        today_data_complete = False
        multi_dayplus_check[1]=False
    else:
        today_data_complete = not today_data_check.isna().any().any()

    # 检查日前特征是否有未来5天的数据
    dayahead_data_complete = True
    for i in range(1, 6):  # 检查未来5天
        check_date = (current_datetime + timedelta(days=i)).date()
        check_start = datetime.combine(check_date, time(hour=0, minute=0, second=0))
        check_end = datetime.combine(check_date, time(hour=23, minute=45, second=0))
        dayahead_data_check = new_data.loc[check_start:check_end, d_columns_dict[i]]
        if dayahead_data_check.empty or dayahead_data_check.isna().any().any():
            dayahead_data_complete = False
            multi_dayplus_check[i]=False
            # 不再break，继续检查所有天数

    # 合并新旧数据
    combined_data = new_data.combine_first(existing_data)
    combined_data = combined_data.reindex(columns=columns)
    combined_data.to_parquet(data_dir)


    # 检查今日d+1数据和昨日d+2数据是否相同，检测数据源是否用昨日的d+2数据填充了今日的d+1数据
    check_date_d1 = (current_datetime + timedelta(days=1)).date()
    check_date_d2_yesterday = (current_datetime - timedelta(days=1) + timedelta(days=2)).date()  # 昨日的d+2
    check_start_d1 = datetime.combine(check_date_d1, time(hour=0, minute=0, second=0))
    check_end_d1 = datetime.combine(check_date_d1, time(hour=23, minute=45, second=0))
    check_start_d2_yesterday = datetime.combine(check_date_d2_yesterday, time(hour=0, minute=0, second=0))
    check_end_d2_yesterday = datetime.combine(check_date_d2_yesterday, time(hour=23, minute=45, second=0))
    
    # 获取今日d+1和昨日d+2的数据
    data_d1 = new_data.loc[check_start_d1:check_end_d1, d_columns_dict[1]]
    data_d2_yesterday = new_data.loc[check_start_d2_yesterday:check_end_d2_yesterday, d_columns_dict[2]]
    
    # 检查数据是否相同
    data_identical = False
    identical_columns = []
    if not data_d1.empty and not data_d2_yesterday.empty:
        # 比较关键列的数据
        key_columns = ["非市场化机组出力-日前(MW)_d1", "联络线计划-日前(MW)_d1", "竞价空间-日前(MW)_d1", "省调负荷-日前(MW)_d1", 
                      "新能源负荷-日前(MW)_d1", "日前光伏(MW)_d1", "日前风电(MW)_d1"]
        key_columns_d2 = ["非市场化机组出力-日前(MW)_d2", "联络线计划-日前(MW)_d2", "竞价空间-日前(MW)_d2", "省调负荷-日前(MW)_d2",
                           "新能源负荷-日前(MW)_d2", "日前光伏(MW)_d2", "日前风电(MW)_d2"]
        
        data_identical = True
        for col1, col2 in zip(key_columns, key_columns_d2):
            if col1 in data_d1.columns and col2 in data_d2_yesterday.columns:
                if data_d1[col1].equals(data_d2_yesterday[col2]):
                    identical_columns.append(col1.replace("_d1", ""))
                else:
                    data_identical = False
        
        if data_identical:

            print("警告：今日d+1和昨日d+2数据完全相同，可能存在数据源问题")
            print(f"相同的列：{', '.join(identical_columns)}")
            print("可能原因：数据源暂时未更新，用昨日的d+2数据填充了今日的d+1数据")
            print("建议：等待数据源更新（通常半小时内），d+1数据会替换为正确的预测值")
        else:
            print("今日d+1和昨日d+2数据检查正常，数据不同")
            if identical_columns:
                print(f"部分列相同：{', '.join(identical_columns)}")
    else:
        print("无法进行今日d+1和昨日d+2数据比较，数据为空")

    # 输出
    if intraday_data_complete and today_data_complete and dayahead_data_complete and not data_identical: 
        print("所有数据检查通过")
        result = multi_dayplus_check
    else:
        print("数据检查未通过，请检查数据完整性")
        if data_identical:
            multi_dayplus_check[1]=False
            multi_dayplus_check[2]=False
            print("d+1和d+2数据相同，需要等待数据源更新")
        if not intraday_data_complete:
            multi_dayplus_check[1]=False
            missing_intraday = intraday_data_check.columns[intraday_data_check.isna().any()].tolist()
            print(f"日内特征缺少昨日数据，缺失列：{missing_intraday}")
        if not dayahead_data_complete:
            for dayplus in range(1,6):
                # 只检查那些真正有问题的天数
                if not multi_dayplus_check[dayplus]:
                    check_date = (current_datetime + timedelta(days=dayplus)).date()
                    check_start = datetime.combine(check_date, time(hour=0, minute=0, second=0))
                    check_end = datetime.combine(check_date, time(hour=23, minute=45, second=0))
                    dayahead_data_check = new_data.loc[check_start:check_end, d_columns_dict[dayplus]]
                    missing_dayahead = dayahead_data_check.columns[dayahead_data_check.isna().any()].tolist()
                    print(f"dayplus={dayplus}日前特征缺少最新{dayplus}天数据，缺失列：{missing_dayahead}")
        if not today_data_complete:
            missing_today = today_data_check.columns[today_data_check.isna().any()].tolist()
            if 'price' in missing_today:
                for i in range(1,6):
                    multi_dayplus_check[i]=False
            print(f"今日特征缺少今日数据，缺失列：{missing_today}")
        result = multi_dayplus_check
    
    # 保存dayplus状态到文件，供后续步骤使用
    # status_dir = "./temp"
    # os.makedirs(status_dir, exist_ok=True)
    # status_file = os.path.join(status_dir, "dayplus_status.json")
    
    # status_data = {
    #     "timestamp": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
    #     "dayplus_status": result,
    #     "valid_days": [k for k, v in result.items() if v],
    #     "invalid_days": [k for k, v in result.items() if not v]
    # }
    
    # with open(status_file, 'w', encoding='utf-8') as f:
    #     json.dump(status_data, f, ensure_ascii=False, indent=2)
    
    # print(f"Dayplus状态已保存到: {status_file}")
    # print(f"可用天数: {status_data['valid_days']}")
    # print(f"不可用天数: {status_data['invalid_days']}")
    
    return result

def main(data_dir = "data/market_Dn_data.parquet"):
    START = TIME.time()
    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_datetime = datetime.now().astimezone(beijing_tz)
    print("当天日期：", current_datetime)
    columns = [
        'price',
        # 日内数据
        '实时价格', '非市场化机组出力-日内(MW)', '联络线计划-日内(MW)', '竞价空间-日内(MW)', '省调负荷-日内(MW)',
        '新能源负荷-日内(MW)', '实时风电(MW)', '实时光伏(MW)', '水电出力值-日内(MW)', '实际上旋备用(MW)',
        '实际下旋备用(MW)', '频率实际值(MW)', '实时在线机组容量(MW)', '实时负荷率(%)',
        # 日前在线机组容量
        '日前在线机组容量(MW)', '日前负荷率(%)',
        # 日前d1
        '非市场化机组出力-日前(MW)_d1', '联络线计划-日前(MW)_d1', '竞价空间-日前(MW)_d1', '省调负荷-日前(MW)_d1',
        '新能源负荷-日前(MW)_d1', '正备用-日前(MW)_d1', '新能源负荷备用-日前(MW)_d1', '日前光伏(MW)_d1', '日前风电(MW)_d1',
        # 日前d2
        '非市场化机组出力-日前(MW)_d2', '联络线计划-日前(MW)_d2', '竞价空间-日前(MW)_d2', '省调负荷-日前(MW)_d2',
        '新能源负荷-日前(MW)_d2', '正备用-日前(MW)_d2', '新能源负荷备用-日前(MW)_d2', '日前光伏(MW)_d2', '日前风电(MW)_d2',
        # 日前d3
        '非市场化机组出力-日前(MW)_d3', '联络线计划-日前(MW)_d3', '竞价空间-日前(MW)_d3', '省调负荷-日前(MW)_d3',
        '新能源负荷-日前(MW)_d3', '正备用-日前(MW)_d3', '新能源负荷备用-日前(MW)_d3', '日前光伏(MW)_d3', '日前风电(MW)_d3',
        # 日前d4
        '日前光伏(MW)_d4', '日前风电(MW)_d4',
        # 日前d5
        '日前光伏(MW)_d5', '日前风电(MW)_d5'
    ]
    print('列名数量：', len(columns))
    # 读取现有的parquet文件
    existing_data = pd.read_parquet(data_dir)
    
    # 获取最后一天的日期
    last_date = existing_data.index[-1].date()
    
    # 计算开始更新的日期（最后一天的前两天）
    # 可以手动修改这里选择需要获取的时间，用来填补缺失部分
    start_date = (last_date - timedelta(days=8)).strftime("%Y-%m-%d 00:00:00")
    # start_date = '2025-05-26 00:00:00'

    # 加载数据
    new_data = load_all_data(start_date=start_date, current_datetime=current_datetime, columns=columns)
    
    # 检查数据并上传
    multi_dayplus_check = check_and_upload_data(new_data, existing_data, start_date, current_datetime, columns, data_dir, write_flag=False)
    
    # 判断是否有任何天数可用
    has_valid_days = any(multi_dayplus_check.values()) if isinstance(multi_dayplus_check, dict) else False
    all_days_valid = all(multi_dayplus_check.values()) if isinstance(multi_dayplus_check, dict) else False
    
    if all_days_valid:
        print('所有数据获取成功，耗时：', TIME.time() - START, "秒")
        return True
    elif has_valid_days:
        valid_days = [f"d+{k}" for k, v in multi_dayplus_check.items() if v]
        print(f'部分数据获取成功，可用天数: {", ".join(valid_days)}，耗时：', TIME.time() - START, "秒")
        return True
    else:
        print('数据获取完全失败，所有天数均不可用')
        # 确保在数据检查失败时返回非零退出码
        sys.exit(1)

if __name__ == "__main__":
    main(data_dir = "data/market_Dn_data.parquet")
