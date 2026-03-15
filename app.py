import os
from io import StringIO

import pandas as pd
import streamlit as st

from recommender import load_artifacts, recommend, gemini_explain


st.set_page_config(page_title="Board Game Recommender", layout="wide")


@st.cache_resource
def load():
    if "TFIDF_URL" in st.secrets:
        os.environ["TFIDF_URL"] = st.secrets["TFIDF_URL"]
    if "SVD_URL" in st.secrets:
        os.environ["SVD_URL"] = st.secrets["SVD_URL"]
    if "NORM_URL" in st.secrets:
        os.environ["NORM_URL"] = st.secrets["NORM_URL"]
    return load_artifacts("artifacts")


@st.cache_data(show_spinner=False)
def cached_gemini_explain(api_key: str, recs_csv: str, seed_game: str = "", user_query: str = "") -> pd.DataFrame:
    rec_df = pd.read_csv(StringIO(recs_csv))
    return gemini_explain(
        api_key=api_key,
        rec_df=rec_df,
        seed_game=seed_game or None,
        user_query=user_query or None,
    )


art = load()
df = art.df

st.markdown(
    """
    <style>
        .main-wrap {
            max-width: 1100px;
            margin: 0 auto;
        }

        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px rgba(79, 70, 229, 0.18);
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #e8eaf2;
            border-radius: 22px;
            padding: 1rem 1rem 0.25rem 1rem;
            box-shadow: 0 6px 20px rgba(17, 24, 39, 0.05);
            margin-bottom: 1.4rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.3rem 0 0.9rem 0;
            color: #111827;
        }

        .rec-card {
            background: #ffffff;
            border: 1px solid #e7eaf3;
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 4px 16px rgba(17, 24, 39, 0.05);
        }

        .rec-number {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .rec-title {
            font-size: 1.14rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.45rem;
        }

        .rec-reason {
            color: #374151;
            line-height: 1.6;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
            font-size: 0.98rem;
        }

        .mode-note {
            color: #6b7280;
            font-size: 0.92rem;
            margin-top: 0.1rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎲 Board Game Recommender</div>
        <div class="hero-subtitle">
            Find similar games from review text, either by choosing a seed game or describing what you want in plain English.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Search</div>', unsafe_allow_html=True)

mode = st.radio(
    "Search mode",
    ["Game name", "Natural language query"],
    horizontal=True,
)

st.markdown(
    '<div class="mode-note">Choose a title you already like, or describe the type of game you want.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([2.7, 1.2, 1.4])

with col1:
    if mode == "Game name":
        game = st.selectbox(
            "Choose a game",
            sorted(df["game_name"].dropna().unique().tolist()),
            index=0,
        )
        user_query = ""
    else:
        user_query = st.text_area(
            "Describe what you want",
            placeholder="Example: a strategic engine-building game with lots of replayability and low direct conflict",
            height=110,
        )
        game = ""

with col2:
    top_n = st.slider("Recommendations", 3, 12, 5, 1)

with col3:
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

cluster_options = ["All clusters"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
selected_cluster_desc = st.selectbox("Cluster", cluster_options)

cluster_id = None
if selected_cluster_desc != "All clusters":
    desc_to_id = {v: k for k, v in art.cluster_labels.items()}
    cluster_id = desc_to_id[selected_cluster_desc]

st.markdown("</div>", unsafe_allow_html=True)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if run_query:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

    recs = recommend(
        art=art,
        query_type=query_type,
        query_value=query_value,
        sentiment_weight=sentiment_weight,
        cluster_id=cluster_id,
        top_n=top_n,
    )

    if "GEMINI_API_KEY" in st.secrets:
        with st.spinner("Generating recommendation reasons..."):
            reasons_df = cached_gemini_explain(
                api_key=st.secrets["GEMINI_API_KEY"],
                recs_csv=recs.to_csv(index=False),
                seed_game=game,
                user_query=user_query,
            )
    else:
        reasons_df = recs[["game_name"]].copy()
        reasons_df["reason"] = "Add GEMINI_API_KEY in Streamlit Secrets to show recommendation reasons."

    display_df = recs[["game_name"]].merge(reasons_df, on="game_name", how="left")
    display_df["reason"] = display_df["reason"].fillna("No explanation returned.")

    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    for i, row in enumerate(display_df.itertuples(index=False), start=1):
        st.markdown(
            f"""
            <div class="rec-card">
                <div class="rec-number">{i}</div>
                <div class="rec-title">{row.game_name}</div>
                <div class="rec-reason">{row.reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Choose a game or enter a natural language query to see recommendations.")

st.markdown("</div>", unsafe_allow_html=True)
