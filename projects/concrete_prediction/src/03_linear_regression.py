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

df = pd.read_csv("../data/02 processed/cleaned_concrete.csv")
print("数据预览：")
print(df.head())
print("\n列名:", df.columns.tolist()) # 看所有的列名

# 用age 预测 strength
X = df[["age"]]  # 自变量
y = df["strength"] # 应变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 20%的数据用来测试，80%用来训练 随机种子 训练集：用来教模型 测试集：用来检查模型在新数据上的表现

# 建立并训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 用测试集预测
y_pred = model.predict(X_test)

# 评估模型
mae = mean_absolute_error(y_test, y_pred) # 平均绝对误差。预测平均偏离真实值多少， 越小越好
mse = mean_squared_error(y_test, y_pred) # 均方误差 误差平方后的平均值
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) # 决定系数。 表示模型能解释目标值变化的比例

print(f"系数 w:{model.coef_[0]:.4f}") # 线性回归的斜率
print(f"截距 b:{model.intercept_:.4f}") # 截距


plt.figure(figsize=(8, 5))
sns.scatterplot(x="age", y="strength", data=df, s=80, label="真实数据")
plt.plot(df["age"], model.predict(df[["age"]]), color="red", linewidth=2, label="回归线")
plt.title("龄期与强度线回归")
plt.xlabel("龄期")
plt.ylabel("强度")
plt.legend() # 显示图例
plt.tight_layout()
plt.savefig("linear_regression.png")
plt.show()

# 保存模型
joblib.dump(model, "linear_model.pkl")
print("\n模型已保存为 linear_model.pkl")