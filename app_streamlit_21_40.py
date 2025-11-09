
import streamlit as st
import pandas as pd

st.set_page_config(page_title="儒林外史 21–40 回 · 城市詞頻與共現", layout="wide")

st.title("《儒林外史》Ch.21–40：城市詞頻 & 共現網路（原型）")

freq = pd.read_excel("city_frequency_21_40.xlsx")
st.subheader("城市詞頻（每回）")
st.dataframe(freq)

cities = [c for c in freq.columns if c not in ("Chapter","Title")]
city = st.selectbox("選擇城市", cities)
st.bar_chart(freq.set_index("Chapter")[[city]])

st.divider()
st.subheader("共現網路（章級）：邊列表")
edges = pd.read_csv("cooccurrence_edges_21_40.csv")
st.dataframe(edges)

st.divider()
st.subheader("語境摘錄樣例（用於解讀 RQ2）")
ctx = pd.read_excel("city_context_samples_21_40.xlsx")
sel_city = st.multiselect("篩選城市", cities, default=cities[:3])
sel_label = st.multiselect("篩選語境標籤", sorted(ctx["HeuristicLabel"].unique().tolist()))
q = ctx.copy()
if sel_city:
    q = q[q["City"].isin(sel_city)]
if sel_label:
    q = q[q["HeuristicLabel"].isin(sel_label)]
st.write(f"共 {len(q)} 條")
st.dataframe(q[["Chapter","Title","City","HeuristicLabel","Context"]])
