import streamlit as st
from recommender import load_artifacts, recommend

st.set_page_config(page_title="Board Game Recommender", layout="wide")
st.title("🎲 Board Game Recommender")

@st.cache_resource
def load():
    return load_artifacts("artifacts")

art = load()
df = art.df

with st.sidebar:
    st.header("Query")
    mode = st.radio("Mode", ["Game name", "Text description"], index=0)
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)
    top_n = st.slider("Results", 3, 25, 10, 1)

    cluster_filter = st.checkbox("Filter by cluster")
    cluster_id = None
    if cluster_filter:
        cluster_id = st.selectbox("Cluster", sorted(df["cluster"].unique().tolist()))
        st.caption(art.cluster_labels.get(cluster_id, "Unlabeled"))

st.subheader("Recommendations")

if mode == "Game name":
    game = st.selectbox("Choose a game", sorted(df["game_name"].unique().tolist()))
    recs = recommend(art, "game_name", game, sentiment_weight, cluster_id, top_n)
    st.dataframe(recs, use_container_width=True)
else:
    text = st.text_area("Describe what you want", height=120)
    if st.button("Recommend", type="primary"):
        recs = recommend(art, "text_query", text, sentiment_weight, cluster_id, top_n)
        st.dataframe(recs, use_container_width=True)
