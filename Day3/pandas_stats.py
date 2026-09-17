import pandas as pd

def main():
    df = pd.read_csv("concrete.csv", encoding="utf-8-sig")

    print("总行数:", len(df))
    print("平均强度:", df["强度"].mean())
    print("最高强度:", df["强度"].max())
    print("最低强度:", df["强度"].min())

    print("各龄期平均强度:")
    print(df.groupby("龄期")["强度"].mean())

    df_sorted = df.sort_values("强度", ascending=False)
    df_sorted.to_csv("result.csv", index=False)
    print("已保存 result.csv")

if __name__ == "__main__":
    main()