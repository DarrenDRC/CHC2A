import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="Rulin Waishi · Ch.35–55 · TOP5 Cities",
                   layout="wide")

st.title("Rulin Waishi · Chapters 35–55 · Narrative City Explorer (TOP5)")

# ---------- helper ----------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def safe_read(path, label):
    if os.path.exists(path):
        return load_csv(path)
    else:
        st.warning(f"File not found: `{path}`. Please upload {label} below.")
        return None

def compute_totals(freq_df, city_cols):
    totals = freq_df[city_cols].sum().sort_values(ascending=False)
    return totals

def build_network_positions(edges_df):
    """Use networkx spring layout to get 2D positions for Plotly."""
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]),
                   weight=float(r.get("weight", 1)))
    if len(G.nodes) == 0:
        return None, None
    pos = nx.spring_layout(G, k=0.8, seed=7, weight="weight")  # deterministic
    return G, pos

def network_fig(edges_df):
    G, pos = build_network_positions(edges_df)
    if G is None:
        return go.Figure()

    # nodes
    nodes = list(G.nodes())
    x_nodes = [pos[n][0] for n in nodes]
    y_nodes = [pos[n][1] for n in nodes]

    # edges
    x_edges = []
    y_edges = []
    weights = []
    for a, b, data in G.edges(data=True):
        x_edges += [pos[a][0], pos[b][0], None]
        y_edges += [pos[a][1], pos[b][1], None]
        weights.append(float(data.get("weight", 1.0)))

    edge_trace = go.Scatter(
        x=x_edges, y=y_edges,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines"
    )
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(size=22, line=dict(width=1, color="#333"))
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Co-occurrence Network (Chapters 35–55)",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=520,
    )
    return fig

def df_to_csv_download(df, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV", csv, file_name=filename, mime="text/csv")

# ---------- data loading ----------

DEFAULT_FREQ   = "city_frequency_35_55_TOP5.csv"
DEFAULT_CTX    = "city_context_35_55_TOP5.csv"
DEFAULT_EDGES  = "cooccurrence_edges_35_55_TOP5.csv"
DEFAULT_COMAT  = "cooccurrence_matrix_35_55_TOP5.csv"

with st.sidebar:
    st.header("Data Sources")
    freq_file  = st.text_input("Frequency CSV", value=DEFAULT_FREQ)
    ctx_file   = st.text_input("Context CSV",   value=DEFAULT_CTX)
    edges_file = st.text_input("Edges CSV",     value=DEFAULT_EDGES)
    comat_file = st.text_input("Co-matrix CSV", value=DEFAULT_COMAT)

    st.caption("If any file is missing, upload below to override.")
    up_freq  = st.file_uploader("Upload frequency CSV", type=["csv"], key="up_freq")
    up_ctx   = st.file_uploader("Upload context CSV",   type=["csv"], key="up_ctx")
    up_edges = st.file_uploader("Upload edges CSV",     type=["csv"], key="up_edges")
    up_comat = st.file_uploader("Upload matrix CSV",    type=["csv"], key="up_comat")

# prefer uploaded files if present
if up_freq:  freq_df  = pd.read_csv(up_freq)
else:        freq_df  = safe_read(freq_file, "Frequency CSV")

if up_ctx:   ctx_df   = pd.read_csv(up_ctx)
else:        ctx_df   = safe_read(ctx_file, "Context CSV")

if up_edges: edges_df = pd.read_csv(up_edges)
else:        edges_df = safe_read(edges_file, "Edges CSV")

if up_comat: comat_df = pd.read_csv(up_comat, index_col=0)
else:        comat_df = safe_read(comat_file, "Co-matrix CSV")

if freq_df is None or edges_df is None or ctx_df is None:
    st.stop()

# ---------- top summary ----------

st.subheader("Chapters")
chap_min = int(freq_df["Chapter"].min())
chap_max = int(freq_df["Chapter"].max())
st.write(f"Analyzing chapters **{chap_min}–{chap_max}**.")

city_cols = [c for c in freq_df.columns if c not in ("Chapter", "Title")]

colA, colB = st.columns([1,1])
with colA:
    st.markdown("**Total Frequency by City**")
    totals = compute_totals(freq_df, city_cols)
    st.dataframe(totals.rename("Total").to_frame().T if len(totals)==0 else totals.to_frame(name="Total"))
with colB:
    fig = px.bar(totals, title="City Frequency Totals (Ch.35–55)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- per-city timeline ----------

st.subheader("Per-Chapter Frequency")
c = st.selectbox("Select a city", city_cols, index=0)
fig2 = px.bar(freq_df, x="Chapter", y=c, title=f"{c} per Chapter")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- network ----------

st.subheader("Co-occurrence Network (TOP5)")
if "weight" not in edges_df.columns:
    edges_df["weight"] = 1
fig_net = network_fig(edges_df)
st.plotly_chart(fig_net, use_container_width=True)

st.caption("Edges indicate chapters where two cities both appear; weight = number of co-occurring chapters.")

st.divider()

# ---------- context explorer ----------

st.subheader("Context Explorer")
city_pick = st.multiselect("Cities", city_cols, default=city_cols)
kw = st.text_input("Keyword filter (optional, case-sensitive for now)")
q = ctx_df.copy()
if city_pick:
    q = q[q["City"].isin(city_pick)]
if kw:
    q = q[q["Context"].str.contains(kw, na=False)]
st.write(f"{len(q)} rows")
st.dataframe(q[["Chapter","City","Match","Context"]], use_container_width=True, height=350)
df_to_csv_download(q, "context_filtered.csv")

st.divider()

# ---------- raw tables + download ----------

st.subheader("Data Tables")
tab1, tab2, tab3, tab4 = st.tabs(["Frequency", "Edges", "Co-matrix", "Context"])

with tab1:
    st.dataframe(freq_df, use_container_width=True, height=350)
    df_to_csv_download(freq_df, "city_frequency_TOP5.csv")
with tab2:
    st.dataframe(edges_df, use_container_width=True, height=350)
    df_to_csv_download(edges_df, "cooccurrence_edges_TOP5.csv")
with tab3:
    if comat_df is not None:
        st.dataframe(comat_df, use_container_width=True, height=350)
        df_to_csv_download(comat_df, "cooccurrence_matrix_TOP5.csv")
    else:
        st.info("No co-matrix loaded.")
with tab4:
    st.dataframe(ctx_df, use_container_width=True, height=350)
    df_to_csv_download(ctx_df, "city_context_TOP5.csv")

st.write("---")
st.caption("Rulin Waishi · Chapters 35–55 · Yangzhou / Suzhou / Hangzhou / Nanjing / Beijing")
