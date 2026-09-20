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
strength_dist.png、
age_strength_box.png、
age_strength_scatter.png、
correlation_heatmap.png


# Day6 线性回归预测混凝土强度

## 内容
- 训练/测试集划分
- 线性回归模型训练预测
- MAE,RMSE,R2评估
- 可视化回归线，保存模型

## 运行
python linear_regression.py

## 输出
linear_regression.png、
linear_model.pkl

# Day7 多特征线性回归-混凝土强度预测

## 数据集
UCI Concrete Compressive Strength(1030 样本，8 特征)

## 内容
- 8 特征多变量线性回归
- 数据清洗:缺失、重复、异常
- 模型评估:MAE / RMSE / R2
- 特征系数分析
- 可视化：预测 vs 真实、特征重要性

## 运行
python mutlti_feature_regression.py

## 输出
pred_vs_true.png、
multi_feature_regression.png、
multi_linear_model.pkl

# Day8 模型对比与调参

## 内容
- 线性回归 / Ridge / 随机森林 / XGboost 对比
- 5折交叉验证
- XGBoost网格搜索调参
- 特征重要性分析
- 最佳模型保存

## 运行
python model_comparision.py、
python tune_model.py

## 结束
线性回归 R² 约 0.6，XGBoost 调参后 R² 约 0.9+