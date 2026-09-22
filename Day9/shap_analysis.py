import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
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

# 加载Day8保存的模型
model = joblib.load("../Day8/best_concrete_model.pkl")

# 计算Ship值
X_sample = X.sample(200, random_state=42)  # 抽样加速

explainer = shap.TreeExplainer(model)  # 专门解释树模型
shap_values = explainer.shap_values(X_sample) # 每个样本、每个特征的贡献值
print("Shap值的形状:", shap_values.shape)

# 全局特征重要性
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight") # dpi=150 清晰度
plt.show()
# 每一行是一个特征。
# 每个点是一条样本。
# 横轴是 SHAP 值：右边表示推高强度，左边表示拉低强度。
# 颜色表示特征值高低，通常红色是高值，蓝色是低值。
# 比如水泥这一行，如果红点集中在右边，说明水泥越高，越容易推高强度。

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# 单样本解释
idx = 0
print(f"\n单样本解释(第{idx}条):")
print("特征值：")
print(X_sample.iloc(idx)) # 打印这条样本的 11 个特征值
print(f"预测强度:{model.predict(X_sample.iloc[[idx]])[0]:.2f} MPa")

shap.force_plot(explainer.expected_value, shap_values[idx], X_sample.iloc[idx], feature_names=feature_cols,
                matplotlib=True, show=False)
# explainer.expected_value：基准值，可以理解为模型在所有样本上的平均预测值。
# 用 matplotlib 来画，而不是 SHAP 自带的 JavaScript 图，这样方便保存成 PNG。
plt.tight_layout()
plt.savefig("shap_force.png", dpi=150, bbox_inches="tight")
plt.show()

# 水泥用量 vs 强度大小
plt.figure()
shap.dependence_plot(
    "cement", shap_values, X_sample, 
    feature_names=feature_cols, show=False
)
plt.tight_layout()
plt.savefig("shap_dependence_cement.png", dpi=150, bbox_inches="tight")
plt.show()