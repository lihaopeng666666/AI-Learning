# import pandas as pd
# df = pd.read_excel("Concrete_Data.xls")
# print(df.head())
# print(df.columns.tolist())
# print(df.shape) # 行列

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读数据
df = pd.read_excel("Concrete_Data.xls")
print("原始形状：", df.shape)
print("原始列名：", df.columns.tolist())

# 重命名列
df.columns = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age", "strength"
]

print("\n重命名后:", df.head())
print("\n描述统计:", df.describe())

# 数据清洗

# 返回缺失值
print(df.isnull().sum())

# 检查重复值
print("\n重复行数:", df.duplicated().sum())
df = df.drop_duplicates()

# 检查异常值 强度必须大于0
print("\n强度小于零的行数:", (df["strength"] <= 0).sum())
df = df[df["strength"] > 0]

df = df.reset_index(drop=True)  # 充值索引，丢掉就索引，生成从0开始的新索引
print("\n清洗后形状:", df.shape) 

# 定义特征与目标
feature_cols = ["cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age"
]
X = df[feature_cols]
y = df["strength"]

# 划分训练集和测试集 80%用于训练 20%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)
# strength = 截距 + 系数1 * cement + 系数2 * slag + ... + 系数8 * age

# 预测与评估
y_pred = model.predict(X_test)

# 测试集特征预测强度
mae = mean_absolute_error(y_test, y_pred)
rmse = np.square(mean_absolute_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 查看特征系数
coef_df = pd.DataFrame({
    "特征": feature_cols,
    "系数": model.coef_
}).sort_values("系数", key=abs, ascending=False)
# print(type(model.coef_))

# 可视化
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label="理想预测线")
plt.xlabel("真实强度")
plt.ylabel("预测强度")
plt.title(f"多特征线性回归: 真实 vs 预测 (R²={r2:.3f})")
plt.legend() # 显示图例

plt.tight_layout()
plt.savefig("pred_vs_true.png")
plt.show()

# 特征系数条形图
plt.figure(figsize=(10, 5))
colors = ["red" if c > 0 else "blue" for c in coef_df["系数"]]

plt.barh(coef_df["特征"], coef_df["系数"], color=colors)
plt.xlabel("系数值")
plt.title("各特征值对强度的影响(红色=正相关 蓝色=负相关)")
plt.tight_layout()
plt.savefig("feature_coefficients.png")
plt.show()

# 保存模型
joblib.dump(model, "multi_linear_model.pkl")
print("\n模型已保存为 multi_linear_model.pkl")

# 预测新样本
new_sample = pd.DataFrame([{
    "cement": 540, "slag": 0, "fly_ash": 0, "water": 162,
    "superplasticizer": 2.5, "coarse_agg": 1040,
    "fine_agg": 676, "age": 28
}])
pred = model.predict(new_sample)
print(f"\n新样本的预测强度是{pred[0]:.2f} MPa")

# print("训练集 R²:", model.score(X_train, y_train))
# print("测试集 R²:", model.score(X_test, y_test))