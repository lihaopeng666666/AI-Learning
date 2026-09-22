import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_excel("../Day7/Concrete_Data.xls")
df.columns = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age", "strength"
]

df = df.drop_duplicates()
df = df[df["strength"] > 0].reset_index(drop=True)

# 特征工程
df["water_cement_ratio"] = df["water"] / df["cement"]
df["log_age"] = np.log1p(df["age"]) # 对龄期取对数。因为龄期从几天到几百天，跨度很大，取对数后更平滑。
df["total_agg"] = df["coarse_agg"] + df["fine_agg"] # 总骨料

feature_cols = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age",
    "water_cement_ratio", "log_age", "total_agg"
]
X = df[feature_cols]
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 网格搜索
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}
#  1、XGBoost里树的数量 2、每棵树的最大深度 3、学习率，每棵树对最终结果的贡献大小 4、每棵树训练时用多少比例的样本

xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0) # n_jobs用所有CPU核心加速，verbosity=0 不打印多余信息
grid = GridSearchCV(xgb, param_grid, cv=3, scoring="r2",  n_jobs=-1, verbose=1)
# 从 param_grid 里拿出一种参数组合，用这种组合训练模型。verbose=1 打印搜索进度
# cv = 3 表示 3 折交叉验证：把训练数据分成 3 份，轮流用其中 2 份训练、1 份验证，做 3 次，取平均
# 把r2作为平均机制

grid.fit(X_train, y_train) # 开始搜索

print("最佳参数:", grid.best_params_)
print("最佳交叉验证R²:", grid.best_score_)

# 拿到最佳模型
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(f"\n测试集 MAE:{mean_absolute_error(y_test, y_pred):.2f}")
print(f"\n测试集 RMSE:{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"\n测试集 R²:{r2_score(y_test, y_pred):.4f}")

# 特征重要性
importance = pd.DataFrame({
    "特征": feature_cols,
    "重要性": best_model.feature_importances_  #  XGBoost 模型自己算出来的“特征重要性”
}).sort_values("重要性", ascending=False)

print("\n特征重要性:") 
print(importance.to_string(index=False))  # 把表格打印出来 不显示行号

plt.figure(figsize=(10, 6))
sns.barplot(x="重要性", y="特征", data=importance)
plt.title("XGBoost 特征重要性")
plt.tight_layout()
plt.savefig("feature_importance_png")
plt.show()

# 预测 vs 真实
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], "r--", linewidth=2, label="理想预测线")
plt.xlabel("真实强度")
plt.ylabel("预测强度")
plt.title(f"XGBoost 预测 vs 真实 (R²={r2_score(y_test, y_pred):.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_true_xgb.png")
plt.show()

# 保存模型
joblib.dump(best_model, "best_concrete_model.pkl")
print("\n最佳模型已保存为 best_concrete_model.pkl")