import os
from io import StringIO
import html
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

# 1. Page Config - Using "Wide" but controlling max-width via CSS
st.set_page_config(
    page_title="Board Game Index",
    page_icon="📋",
    layout="wide",
)

# 2. High-Density / High-Readability CSS
st.markdown("""
    <style>
        /* Force a clean, professional sans-serif stack */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700&family=Roboto:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
            color: #1a1a1a;
        }

        /* Tighten up the Streamlit padding defaults */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 1400px;
        }

        /* Header / Dashboard Bar */
        .top-bar {
            background-color: #f1f5f9;
            padding: 0.75rem 1.5rem;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 8px;
        }

        .top-bar h1 {
            font-family: 'Roboto Condensed', sans-serif;
            font-size: 1.5rem;
            margin: 0;
            font-weight: 700;
            color: #0f172a;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Recommendation Grid - Tighter Rows */
        .rec-row {
            display: grid;
            grid-template-columns: 40px 1fr 2fr 180px;
            gap: 15px;
            padding: 10px 15px;
            border-bottom: 1px solid #f1f5f9;
            align-items: center;
            background: white;
        }

        .rec-row:hover {
            background-color: #f8fafc;
        }

        .rec-rank {
            font-family: 'Roboto Condensed', sans-serif;
            font-weight: 700;
            color: #94a3b8;
            font-size: 1.1rem;
        }

        .rec-name {
            font-weight: 700;
            font-size: 1rem;
            color: #0f172a;
        }

        .rec-reason {
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.4;
            padding-right: 10px;
        }

        .rec-tag {
            font-family: 'Roboto Condensed', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            background: #e2e8f0;
            padding: 2px 8px;
            border-radius: 4px;
            text-align: center;
            text-transform: uppercase;
            color: #475569;
        }

        /* Compact Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }
        
        .stSelectbox, .stSlider, .stTextArea {
            margin-bottom: -10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Data Core
@st.cache_resource
def load():
    if "SVD_URL" in st.secrets:
        os.environ["SVD_URL"] = st.secrets["SVD_URL"]
    return load_artifacts("artifacts")

@st.cache_data(show_spinner=False)
def cached_gemini_explain(api_key: str, recs_csv: str, seed_game: str = "", user_query: str = "") -> pd.DataFrame:
    rec_df = pd.read_csv(StringIO(recs_csv))
    return gemini_explain(api_key=api_key, rec_df=rec_df, seed_game=seed_game or None, user_query=user_query or None)

art = load()
all_games = sorted(art.df["game_name"].dropna().unique().tolist())
cluster_options = ["All Genres"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
desc_to_id = {v: k for k, v in art.cluster_labels.items()}

# 4. Sidebar Nav (Condensed)
with st.sidebar:
    st.subheader("Filter Engine")
    mode = st.radio("Search Mode", ["Title Match", "Vibe Match"], horizontal=True)

    if mode == "Title Match":
        game = st.selectbox("Select Game", all_games)
        user_query = ""
    else:
        user_query = st.text_area("Describe gameplay...", height=100)
        game = ""

    st.divider()
    top_n = st.number_input("Count", 5, 20, 10)
    sentiment_weight = st.slider("Vibe Factor", 0.0, 1.0, 0.2)
    selected_cluster = st.selectbox("Genre Filter", cluster_options)
    cluster_id = desc_to_id[selected_cluster] if selected_cluster != "All Genres" else None

# 5. Main Layout
st.markdown(f"""
    <div class="top-bar">
        <h1>Board Game Discovery Index</h1>
        <div style="font-size: 0.8rem; color: #64748b;">TOTAL DATABASE: {len(art.df):,} GAMES</div>
    </div>
""", unsafe_allow_html=True)

run_query = (mode == "Title Match" and game) or (mode == "Vibe Match" and user_query.strip())

if not run_query:
    st.write("Please provide an input in the sidebar to generate recommendations.")
else:
    q_type = "game_name" if mode == "Title Match" else "text_query"
    q_val = game if mode == "Title Match" else user_query.strip()

    try:
        with st.spinner("Loading results..."):
            recs = recommend(art, q_type, q_val, sentiment_weight, cluster_id, top_n)

        if "GEMINI_API_KEY" in st.secrets:
            reasons_df = cached_gemini_explain(
                st.secrets["GEMINI_API_KEY"], recs.to_csv(index=False), game, user_query.strip()
            )
        else:
            reasons_df = recs[["game_name"]].copy()
            reasons_df["reason"] = "AI reasoning disabled."

        display_df = recs.merge(reasons_df, on="game_name", how="left")

        # Table Header
        st.markdown("""
            <div style="display: grid; grid-template-columns: 40px 1fr 2fr 180px; gap: 15px; padding: 5px 15px; background: #0f172a; color: white; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                <div>#</div>
                <div>Game Title</div>
                <div>Recommendation Reasoning</div>
                <div style="text-align: center;">Genre</div>
            </div>
        """, unsafe_allow_html=True)

        # Recommendation Rows
        for i, row in enumerate(display_df.itertuples(index=False), start=1):
            st.markdown(f"""
                <div class="rec-row">
                    <div class="rec-rank">{i}</div>
                    <div class="rec-name">{html.escape(row.game_name)}</div>
                    <div class="rec-reason">{html.escape(row.reason)}</div>
                    <div class="rec-tag">{html.escape(str(row.cluster_label))}</div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
