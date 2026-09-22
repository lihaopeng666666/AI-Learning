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

# 定义模型
models = {
    "线性回归": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "随机森林": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbosity=0)
}

# 交叉验证对比
print("===== 5折交叉验证 R²(训练集)=====")
results = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2") #把训练数据分成 5 份，每次拿 4 份训练，剩下 1 份验证，轮流 5 次，最后看平均表现
    results.append((name, scores.mean(), scores.std())) # 
    print(f"{name:8s}  R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
print(results)

# 测试集上评估
test_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    test_results.append((name, mae, rmse, r2))
    print(f"{name:8s} MAE={mae:.2f} RMSE={rmse:.2f}  R²={r2:.4f}")
best_name = max(test_results, key=lambda x: x[3])[0]
print(f"\n测试及最佳模型: {best_name}")