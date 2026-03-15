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
            max-width: 760px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        .app-shell {
            max-width: 680px;
            margin: 0 auto;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.3rem 0 1rem 0;
        }

        .brand {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #111827;
        }

        .brand-sub {
            font-size: 0.92rem;
            color: #6b7280;
            margin-top: 0.15rem;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #eceff3;
            border-radius: 24px;
            padding: 1rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.75rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 750;
            color: #111827;
            margin: 1.1rem 0 0.35rem 0;
            letter-spacing: -0.01em;
        }

        .section-subtitle {
            color: #6b7280;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }

        .context-chip {
            display: inline-block;
            background: #f3f4f6;
            color: #374151;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.9rem;
            margin-bottom: 0.9rem;
            line-height: 1.35;
        }

        .rec-card {
            background: #ffffff;
            border: 1px solid #ebeef3;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
        }

        .rec-header {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            margin-bottom: 0.45rem;
        }

        .rec-index {
            min-width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.88rem;
            margin-top: 0.05rem;
            flex-shrink: 0;
        }

        .rec-game {
            font-size: 1.05rem;
            font-weight: 800;
            color: #111827;
            line-height: 1.3;
            letter-spacing: -0.01em;
        }

        .rec-reason {
            color: #374151;
            line-height: 1.58;
            font-size: 0.96rem;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
            margin-left: 2.55rem;
        }

        .divider-space {
            height: 0.2rem;
        }

        [data-testid="stRadio"] > div {
            gap: 0.5rem;
        }

        [data-testid="stRadio"] label {
            border-radius: 999px !important;
        }

        .stSelectbox label, .stTextArea label, .stSlider label {
            font-weight: 650 !important;
            color: #374151 !important;
        }

        .empty-state {
            background: #fafafa;
            border: 1px dashed #d1d5db;
            border-radius: 20px;
            padding: 1rem;
            color: #6b7280;
            font-size: 0.96rem;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .panel {
                padding: 0.9rem;
                border-radius: 20px;
            }

            .rec-card {
                border-radius: 18px;
                padding: 0.95rem 0.9rem 0.9rem 0.9rem;
            }

            .rec-reason {
                margin-left: 2.35rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="brand">Board Game Recommender</div>
            <div class="brand-sub">Search by game or describe what you want.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Search panel
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Search</div>', unsafe_allow_html=True)

mode = st.radio(
    "Mode",
    ["Game name", "Natural language query"],
    horizontal=True,
    label_visibility="collapsed",
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
        placeholder="Strategic engine-building game with strong replayability and low direct conflict",
        height=110,
    )
    game = ""

col1, col2 = st.columns(2)

with col1:
    top_n = st.slider("Recommendations", 3, 12, 5, 1)

with col2:
    sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

cluster_options = ["All clusters"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
selected_cluster_desc = st.selectbox("Cluster", cluster_options)

cluster_id = None
if selected_cluster_desc != "All clusters":
    desc_to_id = {v: k for k, v in art.cluster_labels.items()}
    cluster_id = desc_to_id[selected_cluster_desc]

st.markdown('</div>', unsafe_allow_html=True)

# Results header
st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if not run_query:
    if mode == "Game name":
        subtitle = "Choose a seed game to generate similar recommendations."
    else:
        subtitle = "Enter a natural-language query to generate recommendations."

    st.markdown(
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="empty-state">Your recommendations will appear here once you run a search.</div>',
        unsafe_allow_html=True,
    )
else:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

    if mode == "Game name":
        context_text = f'Based on "{game}"'
        subtitle = "Similar games with concise reasons."
    else:
        context_text = f'Query: "{user_query}"'
        subtitle = "Games that best match your natural-language search."

    st.markdown(
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="context-chip">{html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )

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
                <div class="rec-card">
                    <div class="rec-header">
                        <div class="rec-index">{i}</div>
                        <div class="rec-game">{game_name}</div>
                    </div>
                    <div class="rec-reason">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(str(e))

st.markdown('</div>', unsafe_allow_html=True)
