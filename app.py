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
            max-width: 880px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.01em;
        }

        .app-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 1.1rem;
        }

        .context-line {
            color: #374151;
            font-size: 0.97rem;
            margin-bottom: 1rem;
        }

        .divider {
            border-top: 1px solid #e5e7eb;
            margin: 0.9rem 0 1.2rem 0;
        }

        .result-row {
            padding: 0.95rem 0 1rem 0;
            border-bottom: 1px solid #f0f2f5;
        }

        .result-title {
            font-size: 1.04rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.28rem;
            line-height: 1.35;
        }

        .result-reason {
            font-size: 0.97rem;
            color: #374151;
            line-height: 1.58;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .empty-note {
            color: #6b7280;
            font-size: 0.96rem;
            margin-top: 0.8rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #eceff3;
            background: #fbfbfc;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 1rem !important;
            font-weight: 700 !important;
            color: #111827;
        }

        section[data-testid="stSidebar"] .stRadio > label,
        section[data-testid="stSidebar"] .stSelectbox > label,
        section[data-testid="stSidebar"] .stTextArea > label,
        section[data-testid="stSidebar"] .stSlider > label {
            font-weight: 600;
            font-size: 0.9rem;
            color: #374151;
        }

        section[data-testid="stSidebar"] .stMarkdown p {
            font-size: 0.88rem;
            color: #6b7280;
        }

        .sidebar-section-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.25rem;
            margin-bottom: 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown('<div class="sidebar-section-label">Search</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["Game name", "Natural language query"],
        index=0,
        label_visibility="visible",
    )

    if mode == "Game name":
        game = st.selectbox(
            "Game",
            sorted(df["game_name"].dropna().unique().tolist()),
        )
        user_query = ""
    else:
        user_query = st.text_area(
            "Query",
            placeholder="Strategic engine-building game with strong replayability and low direct conflict",
            height=110,
        )
        game = ""

    st.markdown('<div class="sidebar-section-label">Filters</div>', unsafe_allow_html=True)

    top_n = st.slider("Recommendations", 3, 12, 5, 1)
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

    cluster_options = ["All clusters"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
    selected_cluster_desc = st.selectbox("Cluster", cluster_options)

    cluster_id = None
    if selected_cluster_desc != "All clusters":
        desc_to_id = {v: k for k, v in art.cluster_labels.items()}
        cluster_id = desc_to_id[selected_cluster_desc]


st.markdown('<div class="app-title">Board Game Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Search by seed game or natural-language description.</div>',
    unsafe_allow_html=True,
)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if not run_query:
    if mode == "Game name":
        context_text = "Select a game in the sidebar to generate recommendations."
    else:
        context_text = "Enter a natural-language query in the sidebar to generate recommendations."

    st.markdown(
        f'<div class="context-line">{html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-note">Recommendations will appear here.</div>',
        unsafe_allow_html=True,
    )
else:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

    if mode == "Game name":
        context_text = f'Based on: "{game}"'
    else:
        context_text = f'Based on your query: "{user_query}"'

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
