import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

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

@st.cache_data(show_spinner=False)
def cached_gemini_explain(api_key: str, seed_game: str, recs_csv: str) -> str:
    import pandas as pd
    from io import StringIO
    rec_df = pd.read_csv(StringIO(recs_csv))
    return gemini_explain(api_key=api_key, seed_game=seed_game, rec_df=rec_df)

st.markdown("### Explanation (Gemini)")
if "GEMINI_API_KEY" not in st.secrets:
    st.info("Add GEMINI_API_KEY in Streamlit Cloud → Manage app → Settings → Secrets to enable explanations.")
else:
    if st.button("Explain these recommendations"):
        with st.spinner("Generating explanation…"):
            # convert to CSV so Streamlit can hash/cache it easily
            explanation = cached_gemini_explain(
                api_key=st.secrets["GEMINI_API_KEY"],
                seed_game=game,              # assumes you're in game-name mode
                recs_csv=recs.to_csv(index=False),
            )
        st.write(explanation)
