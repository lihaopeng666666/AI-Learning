import pandas as pd

df = pd.read_csv("concrete.csv") #保存为 DataFrame

print(df.head()) # 查看前五行
print(df.info()) # 查看表单的基本信息
print(df.describe()) # 数列值的统计信息

print(df["强度"]) # “强度这一列打印” 返回Series
print(df[df["龄期"] == 7]) # 筛选出龄期等于7的所有行
print(df.groupby("龄期")["强度"].mean()) # 按“龄期”分组了，然后计算每个龄期下的“强度”的平均值
print(df.sort_values("强度", ascending=False)) # 按照“强度”列排序 降序
df["强度等级"] = df["强度"].apply(lambda x:"高" if x > 35 else "低") # 新增一列“强度等级”
df.to_csv("result.csv", index=False) # 把处理后的 df保存为result.csv 不把 pandas的行索引引入csv
print(df)