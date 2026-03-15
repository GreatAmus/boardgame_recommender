import os
from io import StringIO
import html

import pandas as pd
import streamlit as st

from recommender import load_artifacts, recommend, gemini_explain


st.set_page_config(
    page_title="Board Game Recommender",
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


# ---------- STYLING ----------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1000px;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .results-wrap {
            margin-top: 0.5rem;
        }

        .results-divider {
            border-top: 1px solid #e5e7eb;
            margin: 1.25rem 0 1.5rem 0;
        }

        .results-subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-top: -0.35rem;
            margin-bottom: 1.25rem;
        }

        .rec-item {
            padding: 0.85rem 0 1rem 0;
            border-bottom: 1px solid #eceff3;
        }

        .rec-line {
            font-size: 1.03rem;
            line-height: 1.65;
            color: #111827;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .rec-num {
            font-weight: 700;
            color: #111827;
        }

        .rec-game {
            font-weight: 700;
            color: #111827;
        }

        .rec-reason {
            color: #374151;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #eceff3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Board Game Recommender")
    st.caption("Search settings")

    mode = st.radio(
        "Search mode",
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

    st.markdown("---")
    st.caption("Recommendations update automatically when you change the search.")


# ---------- MAIN ----------
st.title("Recommendations")

if mode == "Game name":
    st.markdown(
        '<div class="results-subtitle">Choose a seed game in the sidebar to find similar games.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="results-subtitle">Enter a natural language query in the sidebar to find matching games.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="results-divider"></div>', unsafe_allow_html=True)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

if not run_query:
    if mode == "Game name":
        st.info("Pick a game in the sidebar to generate recommendations.")
    else:
        st.info("Enter a description in the sidebar to generate recommendations.")
else:
    query_type = "game_name" if mode == "Game name" else "text_query"
    query_value = game if mode == "Game name" else user_query

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

        if mode == "Game name":
            st.subheader(f'Based on "{game}"')
            st.caption("Recommended games and why they are similar.")
        else:
            st.subheader("Based on your query")
            st.caption(f'"{user_query}"')

        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)

        for i, row in enumerate(display_df.itertuples(index=False), start=1):
            game_name = html.escape(str(row.game_name))
            reason = html.escape(str(row.reason))

            st.markdown(
                f"""
                <div class="rec-item">
                    <div class="rec-line">
                        <span class="rec-num">{i}.</span>
                        <span class="rec-game">{game_name}</span>
                        <span class="rec-reason"> — {reason}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(str(e))
