import os
from io import StringIO
import html

import pandas as pd
import streamlit as st

from recommender import load_artifacts, recommend, gemini_explain


st.set_page_config(
    page_title="Board Game Recommender",
    layout="centered",
)


@st.cache_resource
def load():
    if "SVD_URL" in st.secrets:
        os.environ["SVD_URL"] = st.secrets["SVD_URL"]
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
        .block-container {
            max-width: 860px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-top: -0.35rem;
            margin-bottom: 1.5rem;
        }

        .context-line {
            color: #374151;
            font-size: 0.98rem;
            margin-bottom: 1.25rem;
        }

        .divider {
            border-top: 1px solid #e5e7eb;
            margin: 1rem 0 1.4rem 0;
        }

        .result-row {
            padding: 0.95rem 0 1rem 0;
            border-bottom: 1px solid #f0f2f5;
        }

        .result-title {
            font-size: 1.06rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.3rem;
            line-height: 1.35;
        }

        .result-reason {
            font-size: 0.98rem;
            color: #374151;
            line-height: 1.6;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .empty-note {
            color: #6b7280;
            font-size: 0.98rem;
            margin-top: 1rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #eceff3;
        }

        section[data-testid="stSidebar"] .stRadio > label,
        section[data-testid="stSidebar"] .stSelectbox > label,
        section[data-testid="stSidebar"] .stTextArea > label,
        section[data-testid="stSidebar"] .stSlider > label {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Search")

    mode = st.radio(
        "Mode",
        ["Game name", "Natural language query"],
        index=0,
    )

    if mode == "Game name":
        game = st.selectbox(
            "Choose a game",
            sorted(df["game_name"].dropna().unique().tolist()),
        )
        user_query = ""
    else:
        user_query = st.text_area(
            "Describe what you want",
            placeholder="Example: a strategic engine-building game with strong replayability and low direct conflict",
            height=120,
        )
        game = ""

    st.markdown("---")

    top_n = st.slider("Number of recommendations", 3, 12, 5, 1)
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

    cluster_options = ["All clusters"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
    selected_cluster_desc = st.selectbox("Cluster", cluster_options)

    cluster_id = None
    if selected_cluster_desc != "All clusters":
        desc_to_id = {v: k for k, v in art.cluster_labels.items()}
        cluster_id = desc_to_id[selected_cluster_desc]


st.title("Board Game Recommender")
st.markdown(
    '<div class="page-subtitle">Find recommended games from review-based similarity and concise natural-language explanations.</div>',
    unsafe_allow_html=True,
)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if not run_query:
    if mode == "Game name":
        context_text = "Select a game in the sidebar to see recommendations."
    else:
        context_text = "Enter a natural-language query in the sidebar to see recommendations."

    st.markdown(
        f'<div class="context-line">{html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-note">Your recommendations will appear here.</div>',
        unsafe_allow_html=True,
    )
else:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

    if mode == "Game name":
        context_text = f'Showing recommendations based on: "{game}"'
    else:
        context_text = f'Showing recommendations for: "{user_query}"'

    st.markdown(
        f'<div class="context-line">{html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    try:
        recs = recommend(
            art=art,
            query_type=query_type,
            query_value=query_value,
            sentiment_weight=sentiment_weight,
            cluster_id=cluster_id,
            top_n=top_n,
        )

        if "GEMINI_API_KEY" in st.secrets:
            with st.spinner("Generating explanations..."):
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

        for i, row in enumerate(display_df.itertuples(index=False), start=1):
            game_name = html.escape(str(row.game_name))
            reason = html.escape(str(row.reason))

            st.markdown(
                f"""
                <div class="result-row">
                    <div class="result-title">{i}. {game_name}</div>
                    <div class="result-reason">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(str(e))
