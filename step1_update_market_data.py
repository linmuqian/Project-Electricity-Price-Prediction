from datetime import datetime, timedelta
import time as TIME
import pytz
import pandas as pd
import warnings
import pickle as pkl
import re
# from real_capacity.capacity_new import load_online_capacity
# 忽略指定的警告
warnings.simplefilter("ignore", UserWarning)
import numpy as np

# 与tianyan通信前的准备
from dotenv import load_dotenv
load_dotenv()
from pytsingrocbytelinker import PostgreLinker, DbType, WriteData, DBInfo

# 开始时间


# 获取当前日期和时间
beijing_tz = pytz.timezone('Asia/Shanghai')
current_datetime = datetime.now().astimezone(beijing_tz)+timedelta(days=1)
print("当天日期：", current_datetime)

columns_dict = {
    '实时价格': 'clean_price_real_time',
    'price': 'clean_price_day_ahead',
    '非市场化机组出力-日内(MW)': 'non_market_unit_power_real_time',
    '非市场化机组出力-日前(MW)': 'non_market_unit_power_day_ahead',
    '联络线计划-日前(MW)': 'contact_line_plan_day_ahead',
    '联络线计划-日内(MW)': 'contact_line_real_time',
    '频率实际值(MW)': 'frequency_real_time',
    '省调负荷-日前(MW)': 'uniform_schedule_load_day_ahead',
    '省调负荷-日内(MW)': 'usage_load_real_time',
    '新能源负荷-日前(MW)': 'new_energy_pred_day_ahead',
    '新能源负荷-日内(MW)': 'new_energy_plan_real_time',
    '实时风电(MW)': 'wind_power_real_time',
    '实时光伏(MW)': 'pv_power_real_time',
    '日前风电(MW)': 'wind_power_pred_day_ahead',
    '日前光伏(MW)': 'pv_power_pred_day_ahead',
    '水电出力值-日内(MW)': 'hydro_power_real_time',
    '正备用-日前(MW)': 'standby_p_day_ahead',
    '新能源负荷备用-日前(MW)': 'standby_new_energy_day_ahead',
    '实际上旋备用(MW)': 'standby_up_spinning_real_time',
    '实际下旋备用(MW)': 'standby_down_spinning_real_time',
    '日前在线机组容量(MW)': 'online_unit_cap_day_ahead',
    '实时在线机组容量(MW)': 'online_unit_cap_real_time',
    '实时负荷率(%)': 'load_rate_real_time',
    '竞价空间-日内(MW)': 'bidding_space_real_time',
    '竞价空间-日前(MW)': 'bidding_space_day_ahead',
    '日前负荷率(%)': 'load_rate_day_ahead',
}
def read_intraday_data(start_date="2025-04-16 00:00:00", end_date=None):
    """
    读取日内数据
    """
    end_date = (current_datetime.replace(tzinfo=None) - timedelta(days=1)).replace(
        hour=23, minute=45, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")
    
    intra_columns = [
        "实时价格",
        "非市场化机组出力-日内(MW)",
        "联络线计划-日内(MW)",
        "频率实际值(MW)",
        "实际上旋备用(MW)",
        "实际下旋备用(MW)",
        "省调负荷-日内(MW)",
        "新能源负荷-日内(MW)",
        "实时风电(MW)",
        "实时光伏(MW)",
        "水电出力值-日内(MW)",
        "实时在线机组容量(MW)",
        "实时负荷率(%)",
        "竞价空间-日内(MW)",
    ]
    
    byte_linker = PostgreLinker()
    byte_linker.connect(DbType.test)

    all_data = []
    for column in intra_columns:
        target_tags ={}
        if column =="联络线计划-日内(MW)":
            target_tags ={"channel":"总加"}
        db_info = DBInfo()
        db_info.read_from_data(
            table_name="market_data_shanxi",
            key=columns_dict[column],
            tags=target_tags,
            daypoints=96,
            column_name=column,
            start_time=start_date,
            end_time=end_date,
            region="cn-shanxi",
            value_tag="value",
        )
        data = byte_linker.query(db_info)
        data.set_index('time', inplace=True)
        all_data.append(data)
    
    byte_linker.disconnect()
    return pd.concat(all_data, axis=1)

def read_dayahead_data(start_date="2025-04-16 00:00:00", end_date=None,dayplus=1):
    """
    读取日前数据
    """
    end_date = (current_datetime.replace(tzinfo=None) + timedelta(days=1)).replace(
        hour=23, minute=45, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")
    
    dayahead_columns = [
        'price',
        "非市场化机组出力-日前(MW)",
        "联络线计划-日前(MW)", 
        "省调负荷-日前(MW)",
        "新能源负荷-日前(MW)",
        "正备用-日前(MW)",
        "新能源负荷备用-日前(MW)",
        "日前风电(MW)",
        "日前光伏(MW)",
        '日前在线机组容量(MW)',
        '竞价空间-日前(MW)',
        '日前负荷率(%)',
    ]
    
    byte_linker = PostgreLinker()
    byte_linker.connect(DbType.test)
    
    all_data = []
    for column in dayahead_columns:

        if column in ['price']:
            target_tags={}
        elif column=='联络线计划-日前(MW)':
            target_tags={"channel":"总加","dayplus":dayplus}
        else:
            target_tags={"dayplus":dayplus}

        if column in ["正备用-日前(MW)", "新能源负荷备用-日前(MW)"]:
            start = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start = start_date
            
        db_info = DBInfo()
        db_info.read_from_data(
            table_name="market_data_shanxi",
            key=columns_dict[column],
            tags=target_tags,
            daypoints=96,
            column_name=column,
            start_time=start,
            end_time=end_date,
            region="cn-shanxi",
            value_tag="value" 
        )
        data = byte_linker.query(db_info)
        data.set_index('time', inplace=True)
        # 对于每天一个值的数据，数据库上保存的时间为00:00:00，但是会被接口认为是昨天的数据
        if column in ["正备用-日前(MW)", "新能源负荷备用-日前(MW)"]:
            data.index = data.index + pd.Timedelta(days=1)
            
        all_data.append(data)
    
    byte_linker.disconnect()
    return pd.concat(all_data, axis=1)

def process_combined_data(combined_data, columns):
    """
    处理合并后的数据,目前主要是对每天一值的列进行处理
    """
    # 处理正备用和新能源负荷备用,将每天一个值的列填充到全天
    for column in ["正备用-日前(MW)", "新能源负荷备用-日前(MW)"]:
        daily_values = combined_data[column].resample('D').first()
        for day, value in daily_values.items():
            if pd.notna(value):
                day_start = day.replace(hour=0, minute=0, second=0)
                day_end = day.replace(hour=23, minute=59, second=59)
                combined_data.loc[day_start:day_end, column] = value
    # 计算特征
    # 联络线计划，数据库中读取到的是负值，代表山西向外输电，由于数据库目前未补充至20230101，之前的数据使用的是融一的为正值，所以暂时取反，后续会将负号去掉，然后修改竞价空间计算方法
    combined_data['联络线计划-日前(MW)']=-combined_data['联络线计划-日前(MW)']
    combined_data['联络线计划-日内(MW)']=-combined_data['联络线计划-日内(MW)']
    combined_data.index=combined_data.index.tz_localize(None)
    
    # 计算竞价空间
    combined_data['竞价空间-日前(MW)'] = combined_data['省调负荷-日前(MW)'] + combined_data['联络线计划-日前(MW)'] - combined_data['日前光伏(MW)'] - combined_data['日前风电(MW)'] - combined_data["非市场化机组出力-日前(MW)"]
    combined_data['竞价空间-日内(MW)'] = combined_data['省调负荷-日内(MW)'] + combined_data['联络线计划-日内(MW)'] - combined_data['实时光伏(MW)'] - combined_data['实时风电(MW)'] - combined_data["非市场化机组出力-日内(MW)"]
    #combined_data.loc["2024-09-10 00:00:00",'日前在线机组容量(MW)'] = 36500
    # 计算负荷率
    combined_data['日前负荷率(%)'] = combined_data['竞价空间-日前(MW)'] / combined_data['日前在线机组容量(MW)'] * 100  
    combined_data['实时负荷率(%)'] = combined_data['竞价空间-日内(MW)'] / combined_data['日前在线机组容量(MW)'] * 100
    
    return combined_data.reindex(columns=columns)

def load_all_data(start_date="2025-04-16 00:00:00", end_date=None, columns=None):
    """
    加载所有数据的主函数
    """
    # 读取日内数据
    intraday_data = read_intraday_data(start_date, end_date)
    # 读取日前数据
    dayahead_data = read_dayahead_data(start_date, end_date)
    # 去除重复的时间index行
    intraday_data = intraday_data[~intraday_data.index.duplicated(keep='first')]
    dayahead_data = dayahead_data[~dayahead_data.index.duplicated(keep='first')]
    # # 读取实时在线机组容量数据
    # real_capacity = load_online_capacity(start_date, end_date)
    # 合并日内和日前数据
    combined_data = pd.concat([intraday_data, dayahead_data], axis=1)

    if columns is not None:
        final_data = process_combined_data(combined_data, columns)
        final_data=final_data.reindex(columns=columns)
        return final_data
    return combined_data

def main():
    START = TIME.time()
    columns = ['price','实时价格', '实时在线机组容量(MW)', '实时负荷率(%)','非市场化机组出力-日内(MW)', '联络线计划-日内(MW)', '竞价空间-日内(MW)',
       '频率实际值(MW)', '实际上旋备用(MW)', '实际下旋备用(MW)', '省调负荷-日内(MW)',
       '新能源负荷-日内(MW)','实时风电(MW)','实时光伏(MW)','水电出力值-日内(MW)', '日前在线机组容量(MW)', '日前负荷率(%)',
         '非市场化机组出力-日前(MW)', '联络线计划-日前(MW)', '竞价空间-日前(MW)',
       '省调负荷-日前(MW)', '新能源负荷-日前(MW)', '正备用-日前(MW)', '新能源负荷备用-日前(MW)','日前光伏(MW)','日前风电(MW)']
    
    # 读取现有的parquet文件
    existing_data = pd.read_parquet(rf'data/processed/shanxi_new.parquet')

    #existing_data = existing_data.loc[:"2025-08-05 23:45:00"]
    
    # 获取最后一天的日期
    last_date = existing_data.index[-1].date()
    
    # 计算开始更新的日期（最后一天的前两天）
    # 可以手动修改这里选择需要获取的时间，用来填补缺失部分
    start_date = (last_date - timedelta(days=4)).strftime("%Y-%m-%d 00:00:00")
    # start_date = '2025-04-21 00:00:00'
    # 加载新数据
    new_data = load_all_data(start_date=start_date, columns=columns)
    
    # 检查数据是否有异常
    intraday_columns = ['非市场化机组出力-日内(MW)','竞价空间-日内(MW)','实时价格', '联络线计划-日内(MW)', '频率实际值(MW)', '实际上旋备用(MW)', '实际下旋备用(MW)', 
                        '省调负荷-日内(MW)', '新能源负荷-日内(MW)', '实时风电(MW)', '实时光伏(MW)', '水电出力值-日内(MW)','实时在线机组容量(MW)','实时负荷率(%)']
    dayahead_columns = [ '非市场化机组出力-日前(MW)', '联络线计划-日前(MW)', 
                        '竞价空间-日前(MW)', '省调负荷-日前(MW)', '新能源负荷-日前(MW)', '正备用-日前(MW)', '新能源负荷备用-日前(MW)', 
                        '日前光伏(MW)', '日前风电(MW)']
    today_columns = ['price','日前在线机组容量(MW)', '日前负荷率(%)']


    # 检查日内特征是否有昨日的数据
    yesterday = (datetime.now() - timedelta(days=1)).date()
    yesterday_start = datetime.combine(yesterday, datetime.min.time()).replace(hour=0, minute=0, second=0)  
    yesterday_end = datetime.combine(yesterday, datetime.max.time()).replace(hour=23, minute=45)
    intraday_data_check = new_data.loc[yesterday_start:yesterday_end, intraday_columns]
    if intraday_data_check.empty:
        intraday_data_complete = False
    else:
        intraday_data_complete = not intraday_data_check.isna().any().any()

    # 检查今日特征是否有今日的数据
    today = datetime.now().date()
    today_start = datetime.combine(today, datetime.min.time()).replace(hour=0, minute=0, second=0)
    today_end = datetime.combine(today, datetime.max.time()).replace(hour=23, minute=45)
    today_data_check = new_data.loc[today_start:today_end, today_columns ]
    if today_data_check.empty:
        today_data_complete = False
    else:
        today_data_complete = not today_data_check.isna().any().any()

    # 检查日前特征是否有明日的数据
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    tomorrow_start = datetime.combine(tomorrow, datetime.min.time()).replace(hour=0, minute=0, second=0)
    tomorrow_end = datetime.combine(tomorrow, datetime.max.time()).replace(hour=23, minute=45)
    dayahead_data_check = new_data.loc[tomorrow_start:tomorrow_end, dayahead_columns]
    if dayahead_data_check.empty:
        dayahead_data_complete = False
    else:
        dayahead_data_complete = not dayahead_data_check.isna().any().any()

    # 新增逻辑，可更新数据能够正常更新
    combined_data = new_data.combine_first(existing_data)
    combined_data = combined_data.reindex(columns=columns)
    
    ### 异常处理
    combined_data.loc["2024-09-10",'日前在线机组容量(MW)'] = 36500
    # 计算负荷率
    combined_data['竞价空间-日前(MW)'] = combined_data['省调负荷-日前(MW)'] + combined_data['联络线计划-日前(MW)'] - combined_data['日前光伏(MW)'] - combined_data['日前风电(MW)'] - combined_data["非市场化机组出力-日前(MW)"]
    combined_data['竞价空间-日内(MW)'] = combined_data['省调负荷-日内(MW)'] + combined_data['联络线计划-日内(MW)'] - combined_data['实时光伏(MW)'] - combined_data['实时风电(MW)'] - combined_data["非市场化机组出力-日内(MW)"]
    combined_data['日前负荷率(%)'] = combined_data['竞价空间-日前(MW)'] / combined_data['日前在线机组容量(MW)'] * 100  
    combined_data['实时负荷率(%)'] = combined_data['竞价空间-日内(MW)'] / combined_data['日前在线机组容量(MW)'] * 100

    combined_data.to_parquet(rf'data/processed/shanxi_new.parquet')
    if intraday_data_complete and today_data_complete and dayahead_data_complete:
        print("数据检查通过，可以进行存储")
        # 合并新旧数据
        print("数据更新完成，耗时：", TIME.time() - START, "秒")
        return True
    else:
        print("数据检查未通过，请检查数据完整性")
        if not intraday_data_complete:
            if intraday_data_check.empty:
                print(f"日内特征缺少昨日数据，缺失列：{intraday_columns}")
            else:
                missing_intraday = intraday_data_check.columns[intraday_data_check.isna().any()].tolist()
                print(f"日内特征缺少昨日数据，缺失列：{missing_intraday}")
        if not dayahead_data_complete:
            if dayahead_data_check.empty:
                print(f"日前特征缺少最新1天数据，缺失列：{dayahead_columns}")
            else:
                missing_dayahead = dayahead_data_check.columns[dayahead_data_check.isna().any()].tolist()
                print(f"日前特征缺少最新1天数据，缺失列：{missing_dayahead}")
        if not today_data_complete:
            if today_data_check.empty:
                print(f"今日特征缺少今日数据，缺失列：{today_columns}")
            else:
                missing_today = today_data_check.columns[today_data_check.isna().any()].tolist()
                print(f"今日特征缺少今日数据，缺失列：{missing_today}")
        return False

if __name__ == "__main__":
    main()
