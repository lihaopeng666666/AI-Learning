import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体是黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号是方块的问题

df = pd.read_csv("../data/02 processed/cleaned_concrete.csv")
print("数据预览：")
print(df.head())

print("\n描述统计")
print(df.describe())

# 画强度分布直方图
plt.figure(figsize=(8, 5)) # 创建一个8*5英寸的画布
sns.histplot(df["strength"], bins=10, kde=True) # 把数据分成十个区间，再画一条平滑的密度曲线
plt.title("混凝土强度分布")
plt.xlabel("强度")
plt.ylabel("频数")
plt.tight_layout() # 自动调整分布，防止文字重叠
plt.savefig("strength_dist.png") # 把图保存成图片
plt.show()

# 画龄期与强度箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(x="age", y="strength", data=df)
plt.title("不同龄期的强度箱线图")
plt.xlabel("龄期")
plt.ylabel("强度")
plt.tight_layout()
plt.savefig("age_strength_box.png")
plt.show()

# 画龄期与强度散点图
plt.figure(figsize=(8, 5)) # 创建一个8*5英寸的画布
sns.scatterplot(x="age", y="strength", data=df, s=80) # s=80 代表点的大小
plt.title("龄期与强度分布")
plt.xlabel("龄期")
plt.ylabel("强度")
plt.tight_layout() # 自动调整分布，防止文字重叠
plt.savefig("age_strength_scatter.png") # 把图保存成图片
plt.show()

# 画相关性热力图
plt.figure(figsize=(6, 4))
corr = df[["strength", "age"]].corr() #只取这两列 计算机相关系数矩阵
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f") #annot在格子里显示数值 coolwarm从冷色到暖色 
plt.title("相关性热力图")
plt.tight_layout()
plt.savefig("correlation_heatmap.png") # 把图保存成图片
plt.show()

# 打印相关系数
print("\n相关系数")
print(df[["strength", "age"]].corr())

# 输出结论
print("\nEDA 结论：")
print("1. 强度主要分布在 33~40 之间,且集中在33附近(呈左偏分布)。")
print("2. 28 天龄期强度整体高于 7 天龄期。")
print("3. 龄期与强度呈正相关，相关系数约 0.71 左右。")
print("4. 未发现明显极端异常值。")