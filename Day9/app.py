import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource  # 是缓存：模型只加载一次，之后网页刷新不用反复读取，速度更快。
def load_model():
    return joblib.load("../Day8/best_concrete_model.pkl")

model = load_model()
st.set_page_config(page_title="混凝土强度预测",  page_icon="🏗️", layout="centered")
st.title("🏗️ 混凝土抗压强度预测系统")
st.markdown("输入配合比参数, 预测28天抗压强度(MPa)") # 说明文字

st.sidebar.header("配合比参数") # 表示这些输入控件放在网页左边
cement = st.sidebar.slider("水泥 (kg/m³)", 100, 600, 350) # 最小100，最大600，默认350
slag = st.sidebar.slider("高炉矿渣 (kg/m³)", 0, 400, 0)
fly_ash = st.sidebar.slider("粉煤灰 (kg/m³)", 0, 300, 0)
water = st.sidebar.slider("水 (kg/m³)", 100, 250, 180)
superplasticizer = st.sidebar.slider("减水剂 (kg/m³)", 0, 30, 5)
coarse_agg = st.sidebar.slider("粗骨料 (kg/m³)", 800, 1200, 1000)
fine_agg = st.sidebar.slider("细骨料 (kg/m³)", 500, 900, 700)
age = st.sidebar.slider("龄期 (天)", 1, 365, 28)

input_df = pd.DataFrame([{
    "cement": cement, "slag": slag, "fly_ash": fly_ash,
    "water": water, "superplasticizer": superplasticizer,
    "coarse_agg": coarse_agg, "fine_agg": fine_agg, "age": age,
    "water_cement_ratio": water / cement,
    "log_age": np.log1p(age),
    "total_agg": coarse_agg + fine_agg
}])

if st.button("预测强度", type="primary"):
    pred = model.predict(input_df)[0]
    st.success(f"预测抗压强度:**{pred:.2f} MPa**")

    # 强度等级判断
    if pred < 20:
        st.warning("强度偏低，建议调整配合比")
    elif pred < 40:
        st.info("强度中等，可用于一般结构")
    else:
        st.success("强度较高，可用于承重结构")
    
st.subheader("当前输入参数")
st.dataframe(input_df.T.rename(columns={0: "值"})) # .T转置

st.caption("模型: XGBoost | 数据集: UCI Concrete Compressive Strength") # 说明模型类型和数据来源