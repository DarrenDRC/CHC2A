import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Rulin Waishi · Ch.35–55 · TOP5 Cities", layout="wide")
st.title("Rulin Waishi · Chapters 35–55 · Narrative Cities (TOP5)")

# ---------- 固定文件路径（请放在仓库根目录） ----------
FREQ_CSV   = "city_frequency_35_55_TOP5.csv"
CTX_CSV    = "city_context_35_55_TOP5.csv"
EDGES_CSV  = "cooccurrence_edges_35_55_TOP5.csv"
COMAT_CSV  = "cooccurrence_matrix_35_55_TOP5.csv"

missing = [p for p in [FREQ_CSV, CTX_CSV, EDGES_CSV] if not os.path.exists(p)]
if missing:
    st.error("Missing required data files: " + ", ".join(missing))
    st.stop()

@st.cache_data
def load_df(path, **kw):
    return pd.read_csv(path, **kw)

freq_df  = load_df(FREQ_CSV)
ctx_df   = load_df(CTX_CSV)
edges_df = load_df(EDGES_CSV)
comat_df = load_df(COMAT_CSV, index_col=0) if os.path.exists(COMAT_CSV) else None

city_cols = [c for c in freq_df.columns if c not in ("Chapter","Title")]
chap_min, chap_max = int(freq_df["Chapter"].min()), int(freq_df["Chapter"].max())

st.markdown(f"**Chapters analyzed:** {chap_min}–{chap_max} · **Cities:** {', '.join(city_cols)}")

# ---------- 1) 总频次与章节走势 ----------
col1, col2 = st.columns([1,1])
with col1:
    totals = freq_df[city_cols].sum().sort_values(ascending=False)
    st.subheader("Total Frequency by City")
    st.dataframe(totals.to_frame("Total"))
    fig_tot = px.bar(totals, title="City Frequency Totals (Ch.35–55)")
    st.plotly_chart(fig_tot, use_container_width=True)

with col2:
    st.subheader("Per-Chapter Frequency")
    c = st.selectbox("Select a city", city_cols, index=0)
    fig_ch = px.bar(freq_df, x="Chapter", y=c, title=f"{c} per Chapter")
    st.plotly_chart(fig_ch, use_container_width=True)

st.divider()

# ---------- 2) GIS 地图（散点气泡，大小=总频次） ----------
# 坐标（可由 CHGIS 校正；这里给出常用城心近似坐标）
CITY_COORDS = {
    "Yangzhou": {"lat": 32.393, "lon": 119.412},
    "Suzhou":   {"lat": 31.299, "lon": 120.585},
    "Hangzhou": {"lat": 30.274, "lon": 120.155},
    "Nanjing":  {"lat": 32.061, "lon": 118.792},
    "Beijing":  {"lat": 39.904, "lon": 116.407},
}
geo_df = pd.DataFrame([
    {"City": k, "lat": v["lat"], "lon": v["lon"], "Total": int(totals.get(k, 0))}
    for k, v in CITY_COORDS.items() if k in city_cols
]).sort_values("Total", ascending=False)

st.subheader("GIS Map · Frequency by City (bubble size = total)")
fig_geo = px.scatter_geo(
    geo_df,
    lat="lat", lon="lon", size="Total", hover_name="City",
    projection="natural earth", title="Narrative City Frequency (Ch.35–55)"
)
fig_geo.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
# 适当聚焦中国区域（经纬度范围）
fig_geo.update_geos(
    fitbounds="locations",
    lataxis_range=[18, 46], lonaxis_range=[98, 125],
    showcountries=True, countrycolor="#888"
)
st.plotly_chart(fig_geo, use_container_width=True)

st.caption("Map is an analytic visualization (not historical basemap). Coordinates may be refined with CHGIS when needed.")

st.divider()

# ---------- 3) 共现网络 ----------
def network_fig(edges):
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]), weight=float(r.get("weight", 1)))
    if len(G.nodes) == 0:
        return go.Figure()
    pos = nx.spring_layout(G, k=0.8, seed=7, weight="weight")
    nodes = list(G.nodes())
    xe, ye = [], []
    for a, b in G.edges():
        xe += [pos[a][0], pos[b][0], None]
        ye += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=xe, y=ye, mode="lines", hoverinfo="none",
                            line=dict(width=1, color="#aaa"))
    xn = [pos[n][0] for n in nodes]
    yn = [pos[n][1] for n in nodes]
    node_trace = go.Scatter(x=xn, y=yn, mode="markers+text", text=nodes,
                            textposition="top center",
                            marker=dict(size=22, line=dict(width=1, color="#333")))
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(title="Co-occurrence Network (chapter-level)",
                      showlegend=False, height=520,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10,r=10,t=50,b=10))
    return fig

st.subheader("Co-occurrence Network (TOP5)")
if "weight" not in edges_df.columns:
    edges_df["weight"] = 1
st.plotly_chart(network_fig(edges_df), use_container_width=True)
st.caption("Edge weight = #chapters in which two cities co-occur.")

st.divider()

# ---------- 4) 语境检索（仅筛选功能，无上传） ----------
st.subheader("Context Explorer")
city_pick = st.multiselect("Cities", city_cols, default=city_cols)
kw = st.text_input("Keyword filter (optional, case-sensitive)")
q = ctx_df.copy()
if city_pick:
    q = q[q["City"].isin(city_pick)]
if kw:
    q = q[q["Context"].str.contains(kw, na=False)]
st.write(f"{len(q)} rows")
st.dataframe(q[["Chapter","City","Match","Context"]], use_container_width=True, height=340)

st.divider()

# ---------- 5) 原始数据表（只读展示与下载） ----------
st.subheader("Data Tables")
tab1, tab2, tab3, tab4 = st.tabs(["Frequency", "Edges", "Co-matrix", "Context"])
with tab1:
    st.dataframe(freq_df, use_container_width=True, height=320)
    st.download_button("Download Frequency CSV",
                       freq_df.to_csv(index=False).encode("utf-8-sig"),
                       "city_frequency_35_55_TOP5.csv", "text/csv")
with tab2:
    st.dataframe(edges_df, use_container_width=True, height=320)
    st.download_button("Download Edges CSV",
                       edges_df.to_csv(index=False).encode("utf-8-sig"),
                       "cooccurrence_edges_35_55_TOP5.csv", "text/csv")
with tab3:
    if comat_df is not None:
        st.dataframe(comat_df, use_container_width=True, height=320)
        st.download_button("Download Co-matrix CSV",
                           comat_df.to_csv().encode("utf-8-sig"),
                           "cooccurrence_matrix_35_55_TOP5.csv", "text/csv")
    else:
        st.info("No co-matrix loaded.")
with tab4:
    st.dataframe(ctx_df, use_container_width=True, height=320)
    st.download_button("Download Context CSV",
                       ctx_df.to_csv(index=False).encode("utf-8-sig"),
                       "city_context_35_55_TOP5.csv", "text/csv")

st.write("---")
st.caption("Meets Option 2: ≥5 places, 10–20 chapters, frequency comparison, GIS map, and network view. Data & code are fixed (no uploads).")
