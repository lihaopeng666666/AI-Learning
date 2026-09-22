# 混凝土抗压强度预测系统

## 项目简介
基于 UCI 混凝土数据集（1030 样本，8 特征），构建端到端机器学习系统，
预测混凝土 28 天抗压强度。完成数据清洗、EDA、特征工程、多模型对比、
超参数调优与 SHAP 可解释性分析，并部署为 Streamlit 交互式 Web 应用。

> 在线体验：[点击访问 Demo](https://your-app.streamlit.app)  <!-- ⚠️ 替换成你部署后的链接，还没部署可以先删掉这行 -->

## 技术栈
- 语言/工具：Python 3.10+, Git, Jupyter
- 数据处理：pandas, numpy
- 机器学习：scikit-learn, XGBoost
- 可解释性：SHAP
- 可视化：matplotlib, seaborn
- 部署：Streamlit, Streamlit Cloud

## 数据集
[UCI Concrete Compressive Strength](https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls)
- 样本数：1030
- 特征：水泥、矿渣、粉煤灰、水、减水剂、粗骨料、细骨料、龄期
- 目标：抗压强度（MPa）

## 项目结构

```text
concrete_prediction/
├── app.py                      # Streamlit 交互式 Web 应用入口
├── src/                        # 核心代码
│   ├── __init__.py
│   ├── 01_clean_data.py        # 数据清洗
│   ├── 02_eda.py               # 探索性数据分析
│   ├── 03_linear_regression.py # 线性回归模型
│   ├── 04_multi_feature_regression.py # 多特征回归
│   ├── 05_model_comparison.py  # 多模型对比
│   ├── 06_tune_model.py        # 超参数调优
│   └── 07_shap_analysis.py     # SHAP 可解释性分析
├── models/
│   └── best_concrete_model.pkl # 训练好的最佳模型
├── data/
│   └── Concrete_Data.xls       # 原始数据集
├── reports/                    # 分析报告与图表
│   ├── figures/                # 生成的图片 (EDA/模型对比/SHAP)
├── README.md                   # 结果展示
└── requirements.txt            # 依赖包

# 1. 克隆项目
git clone https://github.com/lihaopeng666666/AI-Learning

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练与评估（可选，模型已提供）
python src/05_model_comparison.py
python src/06_tune_model.py

# 4. 可解释性分析
python src/07_shap_analysis.py

# 5. 启动 Web 应用
streamlit run app.py