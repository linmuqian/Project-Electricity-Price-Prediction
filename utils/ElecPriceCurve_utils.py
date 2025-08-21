from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def find_y_vectorized(x_list, bin_curve):
    '''
    输入：
    x_list: 一组待查的横坐标，比如负荷率百分数[12.3, 77.8, 45.6]
    bin_curve: pd.Series，索引是pd.Interval，表示分桶的区间
    输出：
    pd.Series，索引是x_list，值是对应的y值
    '''
    index = pd.IntervalIndex(bin_curve.index)
    locs = index.get_indexer(x_list) # 获取x_list中每个值在bin_curve中的位置
    # 处理边界条件
    left_value = bin_curve.iloc[0]
    right_value = bin_curve.iloc[-1]
    # locs == -1 表示不在任何区间内
    # 小于最左区间的x，get_indexer返回-1
    # 大于等于最右区间右端点的x，get_indexer返回-1
    # 需要判断x_list中每个x的位置
    result = []
    for i, x in enumerate(x_list):
        if x < index[0].left:
            result.append(left_value)
        elif x >= index[-1].right:
            result.append(right_value)
        elif locs[i] == -1:
            # 理论上不会出现，但保险起见
            result.append(np.nan)
        else:
            result.append(bin_curve.iloc[locs[i]])
    return pd.Series(result, index=pd.Index(x_list))


def bin_curve_model(load_rate,price, bin_interval=5, max_load_rate=100):
    '''
    把原始逐时序列（load_rate, price）按负荷率分桶（默认 5% 一桶），在每个桶里取电价均值，得到一条“分桶均值曲线”。
    输入：
    load_rate/price： pd.Series，索引是时间戳，值是负荷率百分数和对应的电价。
    bin_interval：int，分桶的间隔，默认 5%。
    max_load_rate：int，负荷率的最大值，默认 100%。
    输出：
    pd.Series：索引是 pd.IntervalIndex（如 [0,5), [5,10), ...），值是该桶内 price 的均值；内部有一段“缺失填补”逻辑确保所有桶都有值。
    '''
    load_rate = load_rate.copy()
    load_rate[load_rate>max_load_rate] = max_load_rate
    load_rate[load_rate<0] = 0
    data = pd.DataFrame({'loadrate': load_rate, 'price': price.copy()})
    # 计算bin的边界
    bins = np.arange(0, max_load_rate + bin_interval, bin_interval)
    # 将负荷率分bin
    data['loadrate_bin'] = pd.cut(data['loadrate'], bins=bins, right=False)
    # 计算每个bin的电价均值
    price_mean_by_bin = data.groupby('loadrate_bin', observed=False)['price'].mean()
    
    def fill_nan_with_conditions(series):
        # 最左侧的 NaN 用 0 填充
        series = series.ffill().fillna(0)
        # 中间的 NaN 用左右非 NaN 值的平均值填充
        nan_indices = series.index[series.isna()]
        for idx in nan_indices:
            left_value = series.iloc[:idx].dropna().iloc[-1]
            right_value = series.iloc[idx + 1:].dropna().iloc[0]
            series[idx] = (left_value + right_value) / 2
        # 最右侧的 NaN 用左边最后一个非 NaN 值填充
        series = series.ffill()
        return series

    # 应用填充逻辑
    price_mean_by_bin = fill_nan_with_conditions(price_mean_by_bin)
    return price_mean_by_bin


def fill_intervals(original_data, start=0, end=100, closed='left'):
    """
    扩展原始数据的区间范围，并填充新增区间的值。
    
    参数:
    original_data (pd.Series): 原始的分桶数据，索引是区间（pd.Interval），值是对应的数值。
    start (int): 区间的起始点。默认为0。
    end (int): 区间的结束点。默认为100。
    closed (str): 区间的闭合方式，'left' 或 'right'。默认为'left'。
    
    返回:
    pd.Series: 扩展并填充后的数据。索引被重建为标准 1% 网格的 IntervalIndex；能精确对上的区间被赋值；开头/结尾两格做了邻值填充；中间的缺口不会自动填（仍可能是 NaN）。
    """
    # 创建完整的区间列表
    intervals = [pd.Interval(i, i + 1, closed=closed) for i in range(start, end)]
    if closed == 'left':
        intervals[-1] = pd.Interval(end - 1, end, closed='left')  # 最后一个区间是闭区间

    # 初始化一个 Series，用 NaN 填充
    series = pd.Series(index=intervals, dtype=float)

    # 将原始数据填充到 Series 中
    for idx, value in original_data.items():
        if idx in series.index:
            series[idx] = value

    # 填充 [start, start+1] 和 [start+1, start+2]：用其右侧第一个非 NaN 值填充
    for i in reversed(range(start, start + 2)):
        if pd.isna(series.iloc[i]):
            series.iloc[i] = series.iloc[i + 1]

    # 填充 [end-2, end-1] 和 [end-1, end]：用其左侧第一个非 NaN 值填充
    for i in range(len(series) - 2, len(series), 1):
        if pd.isna(series.iloc[i]):
            series.iloc[i] = series.iloc[i - 1]

    return series


def interpolate_bin_curve(price_mean_by_bin, new_interval):
    """
    将分桶曲线从原始间隔插值到新的间隔。
    
    参数:
    price_mean_by_bin (pd.Series): 原始分桶曲线数据，索引是负荷率bin的中点，值是对应的均值价格。
    original_interval (float): 原始分桶曲线的间隔。
    new_interval (float): 目标间隔。
    
    返回:
    pd.Series: 插值后的分桶曲线数据，索引是新的负荷率bin的中点，值是对应的插值价格。
    """
    # 提取原始数据点
    x = price_mean_by_bin.index  # 负荷率bin的中点
    x = [(interval.left+interval.right)/2 for interval in x]  # 将索引转换为列表
    x = np.array(x)
    y = price_mean_by_bin.values  # 均值价格
    
    # 创建二次插值函数
    f = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
    
    # 定义新的负荷率bin（间隔为new_interval）
    new_x = np.arange(x.min(), x.max() + new_interval, new_interval)
    # 计算新的数据点
    new_y = f(new_x)
    new_y = np.clip(new_y, 0, 1500)  # 确保价格不为负
    # 创建新的 Series
    new_x = [pd.Interval(int(x-0.5), int(x+0.5), closed='left') for x in new_x]
    new_price_mean_by_bin = pd.Series(new_y, index=new_x)
    new_price_mean_by_bin = fill_intervals(new_price_mean_by_bin, start=0, end=100, closed='left')
    return new_price_mean_by_bin