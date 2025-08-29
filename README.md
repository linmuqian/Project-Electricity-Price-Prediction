# Automatic hiking predict

## 目录

- [Automatic hiking predict](#automatic-hiking-predict)
  - [目录](#目录)
  - [简介](#简介)
  - [安装](#安装)
  - [用法](#用法)
  - [说明](#说明)
    - [预测模式](#预测模式)
    - [回测模式](#回测模式)


## 简介

实时抬价预测

## 安装

```
pip install requirements.txt
```

需要

```
python==3.10
```

## 用法

实时抬价预测
```
python run_automatic.py
```
每日定时抬价预测
```
chmod +x run_scheduler.sh
./run_scheduler.sh start  # 启动（后台常驻）
./run_scheduler.sh status # 看状态
./run_scheduler.sh tail  # 跟日志
./run_scheduler.sh stop  # 停止
```

## 说明
### 预测模式
步骤1:更新D+1日前特征
```
python step1_update_market_data.py
```
步骤2:对历史时间按是否抬价分类
```
python step2_dataset_generate.py
```
步骤3:使用GRU模型，根据日前特征预测D+1是否抬价
```
python step3_train_GRU.py
```
步骤4:根据D+1抬价预测以及前d日量价关系，预测D+1价格走势
```
python step4_price_prediction.py
```
步骤5:从清鹏网站上下载QP数据或者省调数据
```
python step5_update_dplus.py
```
  或者分别运行
  ```
  python update_Dn_market_data.py
  QP_pred.ipynb
  ```
步骤6:使用前d日量价关系，预测D+2至D+5价格走势
```
python step6_predict_dplus.py
```


### 回测模式
对step1和step2生成的数据集初步了解
```
notebook/step1_shanxi_new_read.ipynb
notebook/step2_dataset_generate.ipynb
```
对step3的GRU模型进行回测分析
```
python scripts/train_multi_GRU.py
notebook/step3_polot_backtest_result.ipynb
```
对step4的量价关系和价格走势做回测分析\
分别使用baseline/match/diffusion的方法对D+1至D+5回测分析\
并对比其预测的月均准确率
```
notebook/step4_price_backtest.ipynb
```
对step4预测出的量价关系和价格走势做可视化
```
notebook/step4_price_prediction.ipynb
```

