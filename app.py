import streamlit as st
import pandas as pd
from io import StringIO
from recommender import load_artifacts, recommend, gemini_explain

st.set_page_config(page_title="Board Game Recommender", layout="wide")
st.title("🎲 Board Game Recommender")

@st.cache_resource
def load():
    return load_artifacts("artifacts")

@st.cache_data(show_spinner=False)
def cached_gemini_explain(api_key: str, seed_game: str, recs_csv: str) -> pd.DataFrame:
    rec_df = pd.read_csv(StringIO(recs_csv))
    return gemini_explain(api_key=api_key, seed_game=seed_game, rec_df=rec_df)

art = load()
df = art.df

# ---------- TOP CONTROLS ----------
st.subheader("Choose a game")

col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    game = st.selectbox(
        "Game",
        sorted(df["game_name"].dropna().unique().tolist())
    )

with col2:
    top_n = st.slider("Number of recommendations", 3, 15, 5, 1)

with col3:
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

# Cluster descriptions only
cluster_options = ["All clusters"] + sorted(art.cluster_labels.values())
selected_cluster_desc = st.selectbox("Cluster", cluster_options)

cluster_id = None
if selected_cluster_desc != "All clusters":
    desc_to_id = {v: k for k, v in art.cluster_labels.items()}
    cluster_id = desc_to_id[selected_cluster_desc]

# ---------- RECOMMENDATIONS BELOW ----------
st.subheader("Recommendations")

recs = recommend(
    art=art,
    query_type="game_name",
    query_value=game,
    sentiment_weight=sentiment_weight,
    cluster_id=cluster_id,
    top_n=top_n,
)

# Auto-generate reasons, no extra button
if "GEMINI_API_KEY" in st.secrets:
    with st.spinner("Generating recommendation reasons..."):
        display_df = cached_gemini_explain(
            api_key=st.secrets["GEMINI_API_KEY"],
            seed_game=game,
            recs_csv=recs.to_csv(index=False),
        )
else:
    display_df = recs[["game_name"]].copy()
    display_df["reason"] = "Add GEMINI_API_KEY in Streamlit Secrets to show reasons."

# Make displayed index start at 1
display_df.index = range(1, len(display_df) + 1)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=False
)
