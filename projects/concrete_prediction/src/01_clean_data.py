import pandas as pd

df = pd.read_csv("../data/01 raw/concrete_dirty.csv")
df = df.drop(columns=["备注"])
print("原始数据：")
print(df)
print("\n缺失值统计:")
print(df.isnull().sum()) # 对每一列进行缺失值统计

df = df.drop_duplicates() # 去重 表单里一模一样的行

df["强度"] = pd.to_numeric(df["强度"], errors="coerce")
df["龄期"] = pd.to_numeric(df["龄期"], errors="coerce") # 类型转化 强度、龄期转数字

df = df.dropna(subset=["强度"]) # 删除强度缺失的那行
df["龄期"] = df["龄期"].fillna(df["龄期"].median()) # 龄期缺失用中位数填
df = df[(df["强度"] > 0) & (df["强度"] <= 100)] # 处理异常值：强度合理范围0-100

df = df.rename(columns={"强度":"strength", "龄期": "age"}) # 重命名行列
df = df.reset_index(drop=True) # 重置索引

print("\n清洗后")
print(df)
print("\n清洗后统计")
print(df.describe())

df.to_csv("cleaned_concrete.csv", index=False)
print("\n已保存 cleaned_concrete.csv")