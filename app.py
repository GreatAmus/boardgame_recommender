import os
from io import StringIO
import html

import pandas as pd
import streamlit as st

from recommender import load_artifacts, recommend, gemini_explain


st.set_page_config(
    page_title="Board Game Recommender",
    page_icon="🎲",
    layout="wide",
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

all_games = sorted(df["game_name"].dropna().unique().tolist())
cluster_options = ["All clusters"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
desc_to_id = {v: k for k, v in art.cluster_labels.items()}


st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
            border: 1px solid #dbe7ff;
            border-radius: 18px;
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            color: #1e293b;
            margin-bottom: 0.3rem;
        }

        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 0;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.1rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .context-chip {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            border: 1px solid #c7d2fe;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .rec-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 0.85rem;
        }

        .rec-row {
            display: flex;
            align-items: flex-start;
            gap: 0.9rem;
        }

        .rec-rank {
            min-width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            margin-top: 0.1rem;
        }

        .rec-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.35rem;
        }

        .rec-reason {
            color: #4b5563;
            line-height: 1.55;
            font-size: 0.95rem;
        }

        .empty-state {
            background: #f9fafb;
            border: 1px dashed #d1d5db;
            border-radius: 14px;
            padding: 1rem;
            color: #6b7280;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_result_card(index: int, game_name: str, reason: str) -> None:
    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-row">
                <div class="rec-rank">{index}</div>
                <div>
                    <div class="rec-title">{html.escape(str(game_name))}</div>
                    <div class="rec-reason">{html.escape(str(reason))}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🎲 Board Game Recommender</div>
        <p class="hero-subtitle">
            Find similar games from a title you already like, or describe the kind of game you want in plain English.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose a search mode and refine the recommendations.</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Search mode",
        ["Game name", "Natural language query"],
        horizontal=True,
    )

    if mode == "Game name":
        game = st.selectbox(
            "Game",
            all_games,
            help="Pick a seed game to find similar recommendations.",
        )
        user_query = ""
    else:
        user_query = st.text_area(
            "Describe the kind of game you want",
            placeholder="Strategic engine-building game with strong replayability and low direct conflict",
            height=120,
        )
        game = ""

    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        top_n = st.slider("Recommendations", 3, 12, 5, 1)

    with slider_col2:
        sentiment_weight = st.slider("Sentiment weight", 0.0, 1.0, 0.25, 0.05)

    selected_cluster_desc = st.selectbox("Cluster", cluster_options)

    cluster_id = None
    if selected_cluster_desc != "All clusters":
        cluster_id = desc_to_id[selected_cluster_desc]

    st.markdown("</div>", unsafe_allow_html=True)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)

    if mode == "Game name":
        subtitle = "Similar games based on your selected title."
        context_text = f'Based on "{game}"' if game else "Choose a game to begin"
    else:
        subtitle = "Games that best match your natural-language request."
        context_text = f'Query: "{user_query.strip()}"' if user_query.strip() else "Enter a query to begin"

    st.markdown(
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="context-chip">{html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )

    if not run_query:
        st.markdown(
            """
            <div class="empty-state">
                Recommendations will appear here after you choose a game or enter a natural-language query.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        query_type = "game_name" if mode == "Game name" else "text_query"
        query_value = game if mode == "Game name" else user_query.strip()

        try:
            with st.spinner("Finding recommendations..."):
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
                        user_query=user_query.strip(),
                    )
            else:
                reasons_df = recs[["game_name"]].copy()
                reasons_df["reason"] = "Add GEMINI_API_KEY in Streamlit Secrets to show recommendation reasons."

            display_df = recs[["game_name"]].merge(reasons_df, on="game_name", how="left")
            display_df["reason"] = display_df["reason"].fillna("No explanation returned.")

            for i, row in enumerate(display_df.itertuples(index=False), start=1):
                render_result_card(i, row.game_name, row.reason)

        except Exception as e:
            st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)
