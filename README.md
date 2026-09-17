# Day2 CSV 统计脚本

## 功能
读取混凝土试块 CSV，统计总行数、平均强度、各龄期数量。

## 运行
python stats.py

## 结果
总行数: 5
平均强度: 35.92
各龄期数量: {'7': 3, '28': 2}
# Day3 Git分支与pandas入门

## 内容
- Git 分支创建、合并、回滚
- pandas 读取 CSV、筛选、分组、排序
- 混凝土数据统计脚本

## 运行
python pandas_stats.py

## 结果
输出总行数、平均强度、最高/最低强度、各龄期平均强度，并保存 result.csv
# Day4 数据清洗

## 内容
- 缺失值、重复值、异常值、类型转换
- pandas 清洗混凝土脏数据
- 输出 cleaned_concrete.csv

## 运行
python clean_data.py
# Day5 EDA 探索性数据分析

## 内容
- 强度分布直方图
- 龄期与强度箱线图、散点图
- 相关性热力图
- 结论：龄期与强度正相关

## 运行
python eda.py

## 输出
strength_dist.png
age_strength_box.png
age_strength_scatter.png
correlation_heatmap.png