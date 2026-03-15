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
        :root {
            --bg: #f6f7fb;
            --surface: #ffffff;
            --surface-2: #f8faff;
            --text: #161b26;
            --muted: #697386;
            --border: #e7ebf3;
            --accent: #4f46e5;
            --accent-soft: #eef2ff;
            --accent-2: #7c3aed;
            --shadow: 0 10px 28px rgba(17, 24, 39, 0.06);
            --radius-lg: 22px;
            --radius-md: 16px;
            --radius-sm: 12px;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }

        .stApp {
            background: var(--bg);
        }

        .block-container {
            max-width: 720px;
            padding-top: 0.8rem;
            padding-bottom: 2.2rem;
        }

        .app-shell {
            max-width: 680px;
            margin: 0 auto;
        }

        /* Hide Streamlit's extra top padding feel a bit */
        [data-testid="stHeader"] {
            background: transparent;
        }

        .appbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.1rem 0 0.85rem 0;
        }

        .appbar-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }

        .brand-badge {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.05rem;
            font-weight: 800;
            box-shadow: 0 10px 18px rgba(79, 70, 229, 0.22);
            flex-shrink: 0;
        }

        .brand-wrap {
            min-width: 0;
        }

        .brand-title {
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text);
            line-height: 1.15;
            margin: 0;
        }

        .brand-subtitle {
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.15rem;
            line-height: 1.2;
        }

        .search-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem;
            box-shadow: var(--shadow);
            margin-bottom: 0.9rem;
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.85rem;
        }

        .panel-title {
            font-size: 0.98rem;
            font-weight: 750;
            color: var(--text);
            letter-spacing: -0.01em;
        }

        .panel-caption {
            font-size: 0.8rem;
            color: var(--muted);
        }

        /* Segmented control */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem;
            background: #f2f4f8;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.3rem;
            margin-bottom: 0.9rem;
        }

        div[data-testid="stRadio"] label {
            margin: 0 !important;
            background: transparent !important;
            border: none !important;
            border-radius: 11px !important;
            min-height: 42px;
            display: flex !important;
            align-items: center;
            justify-content: center;
            color: var(--muted) !important;
            font-weight: 650 !important;
            transition: all 0.15s ease;
        }

        div[data-testid="stRadio"] label:has(input:checked) {
            background: var(--surface) !important;
            color: var(--text) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06), 0 0 0 1px rgba(79, 70, 229, 0.08);
        }

        .field-note {
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: -0.15rem;
            margin-bottom: 0.55rem;
        }

        .filters-label {
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
        }

        .stSelectbox label,
        .stTextArea label,
        .stSlider label {
            color: #354052 !important;
            font-weight: 650 !important;
            font-size: 0.9rem !important;
        }

        .stTextArea textarea {
            border-radius: 14px !important;
        }

        .stSelectbox > div > div,
        .stTextArea > div > div,
        .stSlider {
            margin-bottom: 0.15rem;
        }

        .results-header {
            padding: 0.35rem 0 0.1rem 0;
            margin-bottom: 0.35rem;
        }

        .results-title {
            font-size: 1rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.01em;
        }

        .results-subtitle {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.18rem;
            line-height: 1.35;
        }

        .context-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--accent-soft);
            color: #3730a3;
            border: 1px solid #dfe4ff;
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            font-size: 0.86rem;
            font-weight: 600;
            margin: 0.6rem 0 0.95rem 0;
            max-width: 100%;
            line-height: 1.3;
        }

        .empty-state {
            background: var(--surface);
            border: 1px dashed #d8deea;
            border-radius: 18px;
            padding: 1rem;
            color: var(--muted);
            font-size: 0.94rem;
            box-shadow: 0 4px 14px rgba(17, 24, 39, 0.03);
        }

        .rec-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.95rem 0.95rem 0.9rem 0.95rem;
            margin-bottom: 0.72rem;
            box-shadow: 0 8px 22px rgba(17, 24, 39, 0.045);
        }

        .rec-top {
            display: flex;
            align-items: flex-start;
            gap: 0.72rem;
            margin-bottom: 0.35rem;
        }

        .rec-index {
            min-width: 1.95rem;
            height: 1.95rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.86rem;
            font-weight: 800;
            flex-shrink: 0;
            margin-top: 0.04rem;
        }

        .rec-title-wrap {
            min-width: 0;
        }

        .rec-title {
            font-size: 1.01rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1.28;
            letter-spacing: -0.01em;
            margin: 0;
        }

        .rec-reason {
            margin-left: 2.67rem;
            color: #425066;
            font-size: 0.94rem;
            line-height: 1.55;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .stSpinner > div {
            border-top-color: var(--accent) !important;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 0.55rem;
            }

            .search-panel {
                padding: 0.9rem;
            }

            .brand-title {
                font-size: 1.05rem;
            }

            .brand-subtitle {
                font-size: 0.82rem;
            }

            .rec-card {
                padding: 0.9rem 0.9rem 0.85rem 0.9rem;
            }

            .rec-reason {
                margin-left: 2.58rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="appbar">
        <div class="appbar-left">
            <div class="brand-badge">🎲</div>
            <div class="brand-wrap">
                <div class="brand-title">Board Game Recommender</div>
                <div class="brand-subtitle">Find games by title or natural-language search</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="search-panel">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="panel-header">
        <div class="panel-title">Search</div>
        <div class="panel-caption">Review-based recommendations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Search mode",
    ["Game name", "Natural language query"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Game name":
    st.markdown('<div class="field-note">Choose a seed game to find similar titles.</div>', unsafe_allow_html=True)
    game = st.selectbox(
        "Game",
        sorted(df["game_name"].dropna().unique().tolist()),
        label_visibility="collapsed",
        placeholder="Choose a game",
    )
    user_query = ""
else:
    st.markdown('<div class="field-note">Describe the kind of game you want.</div>', unsafe_allow_html=True)
    user_query = st.text_area(
        "Query",
        placeholder="Strategic engine-building game with strong replayability and low direct conflict",
        height=96,
        label_visibility="collapsed",
    )
    game = ""

st.markdown('<div class="filters-label">Filters</div>', unsafe_allow_html=True)

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

st.markdown(
    """
    <div class="results-header">
        <div class="results-title">Recommendations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if not run_query:
    if mode == "Game name":
        subtitle = "Choose a game above to see similar recommendations."
    else:
        subtitle = "Enter a natural-language query above to see matching recommendations."

    st.markdown(
        f'<div class="results-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="empty-state">Recommendations will appear here once you complete a search.</div>',
        unsafe_allow_html=True,
    )
else:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

    if mode == "Game name":
        subtitle = "Similar games with concise recommendation reasons."
        context_text = f'Based on “{game}”'
    else:
        subtitle = "Games that best match your natural-language request."
        context_text = f'Query: “{user_query}”'

    st.markdown(
        f'<div class="results-subtitle">{html.escape(subtitle)}</div>',
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
                    <div class="rec-top">
                        <div class="rec-index">{i}</div>
                        <div class="rec-title-wrap">
                            <div class="rec-title">{game_name}</div>
                        </div>
                    </div>
                    <div class="rec-reason">{reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(str(e))

st.markdown('</div>', unsafe_allow_html=True)
