from dotenv import load_dotenv
load_dotenv()
from pytsingrocbytelinker import PostgreLinker, DbType, WriteData, DBInfo
import pandas as pd
import re
from datetime import datetime, timedelta
import pytz



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
    '负备用-日前(MW)': 'standby_n_day_ahead',
    '新能源负荷备用-日前(MW)': 'standby_new_energy_day_ahead',
    '实际上旋备用(MW)': 'standby_up_spinning_real_time',
    '实际下旋备用(MW)': 'standby_down_spinning_real_time',
    '日前在线机组容量(MW)': 'clean_overview_day_ahead',
    '实时在线机组容量(MW)': 'online_unit_cap_real_time',
}

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
    "实时在线机组容量(MW)"
]
# 只保留 dayplus 对应的 dk 列
d_columns_dict = {
    0: [
    'price',
    '日前在线机组容量(MW)'
    ],
    1: [
        "非市场化机组出力-日前(MW)_d1",
        "联络线计划-日前(MW)_d1",
        # "竞价空间-日前(MW)_d1",
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
        # "竞价空间-日前(MW)_d2",
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
        # "竞价空间-日前(MW)_d3",
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

def read_intraday_data(start_date="2025-04-16 00:00:00", end_date=None):
    """
    读取日内数据
    """

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
        # data = data[~data.index.duplicated(keep="first")]
        all_data.append(data)
    
    byte_linker.disconnect()
    return pd.concat(all_data, axis=1)
def read_dayahead_data(start_date="2025-04-16 00:00:00", end_date=None, dayplus=1, columns=None):
    """
    读取日前数据
    columns: 需要读取的列名列表，为空时读取所有列
    注意：d+1到d+5返回的数据是有后缀的
    """
    byte_linker = PostgreLinker()
    byte_linker.connect(DbType.test)

    all_data = []

    def get_base_column(col):
        # 处理如 "省调负荷-日前(MW)_d2" 这种带 _dN 后缀的列名
        if col.endswith("_d1") or col.endswith("_d2") or col.endswith("_d3") or col.endswith("_d4") or col.endswith("_d5"):
            return col.rsplit("_d", 1)[0]
        return col

    # 是否手动指定列名
    if columns is None:
        target_columns = d_columns_dict[dayplus]
    else:
        target_columns = columns

    for column in target_columns:
        base_column = get_base_column(column)
        # 兼容新旧列名
        if base_column in ['日前在线机组容量(MW)', 'price']:
            target_tags = {}
        elif base_column == '联络线计划-日前(MW)':
            target_tags = {"channel": "总加", "dayplus": dayplus}
        else:
            target_tags = {"dayplus": dayplus}

        # 对于每天一个值的数据，提前一天取数
        if base_column in ["正备用-日前(MW)", "新能源负荷备用-日前(MW)", '日前在线机组容量(MW)']:
            start = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start = start_date

        db_info = DBInfo()
        db_info.read_from_data(
            table_name="market_data_shanxi",
            key=columns_dict[base_column],
            tags=target_tags,
            daypoints=96,
            column_name=base_column,
            start_time=start,
            end_time=end_date,
            region="cn-shanxi",
            value_tag="value" if base_column != '日前在线机组容量(MW)' else 'overview',
        )
        data = byte_linker.query(db_info)
        data.set_index('time', inplace=True)
        data = data[~data.index.duplicated(keep="first")]
        # 对于每天一个值的数据，数据库上保存的时间为00:00:00，但是会被接口认为是昨天的数据
        if base_column in ["正备用-日前(MW)", "新能源负荷备用-日前(MW)", '日前在线机组容量(MW)']:
            data.index = data.index + pd.Timedelta(days=1)
        if base_column == '日前在线机组容量(MW)':
            capacities = []
            for text in data.values:
                text = str(text)
                pattern = r'运行机组容量(\d+\.?\d*?)MW'
                match = re.search(pattern, text)
                if match:
                    capacities.append(float(match.group(1)))
            capacity_data = pd.DataFrame(
                {column: capacities},
                index=data.index
            )
            # d+1到d+5都要带后缀
            capacity_data.columns = [column]
            all_data.append(capacity_data)
            continue
        # 处理带 _dN 后缀的列名
        if column != base_column:
            data.columns = [column]
        else:
            # d+1到d+5都要带后缀
            if any(column.endswith(f"_d{i}") for i in range(1, 6)):
                data.columns = [column]
        all_data.append(data)

    byte_linker.disconnect()
    return pd.concat(all_data, axis=1)

def get_d0_and_dplus_data(start_date, current_datetime):
    """
    获取d0和d+1到d+4的数据，d0为当天，d+1到d+4为未来四天
    Args:
        start_date (str): 开始日期，格式为 "YYYY-MM-DD HH:MM:SS"
        current_datetime (datetime): 当前日期时间
        columns (list): 需要获取的列名列表
        target_tags (dict): 标签信息
        columns_dict (dict): 列名与数据库字段的映射
        byte_linker: 数据库连接对象
    Returns:
        pd.DataFrame: 合并后的d0和d+1-d+4数据
    """
    # d0数据
    end_date = (current_datetime.replace(tzinfo=None) - timedelta(days=1)).replace(
        hour=23, minute=45, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H:%M:%S")
    # 读取日内数据
    d0_data = read_intraday_data(start_date, end_date)
    # d+1到d+4数据
    dplus_data_list = []
    for dayplus in range(0, 6):  # d+1, d+2, d+3, d+4,d+5
        end_date = (current_datetime.replace(tzinfo=None) + timedelta(days=dayplus)).replace(
            hour=23, minute=45, second=0, microsecond=0
        ).strftime("%Y-%m-%d %H:%M:%S")
        dplus = read_dayahead_data(
            start_date, end_date, dayplus=dayplus
        )
        dplus_data_list.append(dplus)
    # 合并
    combined_data = pd.concat([d0_data] + dplus_data_list, axis=1)
    return combined_data
def read_da_price(dayplus, dbtype, date_begin, date_end,version='v1.1.0',source='清鹏',points=96) -> None:
    byte_linker = PostgreLinker()
    if dbtype == 'test':
        byte_linker.connect(DbType.test)
    elif dbtype == 'prod':
        byte_linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    db_info = DBInfo()
    db_info.read_from_data(
        table_name="price_pred_day_ahead",
        key="price_pred_day_ahead",
        tags={ "points":points,"source": source, "dayplus": dayplus},
        daypoints=points,
        version=version,
        column_name="pred",
        start_time=date_begin,
        end_time=date_end,
        region="cn-shanxi",
        value_tag="value",
    )
    data = byte_linker.query(db_info)
    byte_linker.disconnect()
    return data

def read_pred_load(start_date="2025-04-16 00:00:00", end_date=None,key='pred_load', dayplus=1,source='清鹏',dbtype='test'):
    """
    读取日前数据
    columns: 需要读取的列名列表，为空时读取所有列
    注意：d+1到d+5返回的数据是有后缀的
    """
    byte_linker = PostgreLinker()
    if dbtype == 'test':
        byte_linker.connect(DbType.test)
    elif dbtype == 'prod':
        byte_linker.connect(DbType.prod)
    else:
        raise ValueError("dbtype must be 'test' or 'prod'")
    target_tags={"type":"short_term","source":source,"dayplus": dayplus}
    # 是否手动指定列名
    start = start_date
    if end_date is None:
        now=datetime.now(pytz.timezone('Asia/Shanghai'))
        end_date = (now + timedelta(days=dayplus)).replace(hour=23,minute=45,second=0).strftime("%Y-%m-%d %H:%M:%S")
    db_info = DBInfo()
    db_info.read_from_data(
        table_name="load_pred",
        key=key,
        tags=target_tags,
        daypoints=96,
        column_name=source+'_'+key+'_d'+str(dayplus),
        start_time=start,
        end_time=end_date,
        region="cn-shanxi",
        value_tag="value",
        version='v1.0.0'
    )
    data = byte_linker.query(db_info)
    data.set_index('time', inplace=True)
    byte_linker.disconnect()
    return data