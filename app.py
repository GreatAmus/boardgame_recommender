import os
from io import StringIO
import html
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

# 1. Page Config
st.set_page_config(
    page_title="Board Game Pro",
    page_icon="🎯",
    layout="wide",
)

# 2. Ultra-Compact Editorial CSS
st.markdown("""
    <style>
        /* Pro Sans-Serif Stack */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #334155;
        }

        .block-container {
            padding: 1.5rem 3rem !important;
            max-width: 1400px;
        }

        /* Header Area */
        .header-ui {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        /* The Data Table */
        .game-row {
            display: grid;
            grid-template-columns: 50px 220px 1fr 180px 100px;
            gap: 20px;
            align-items: center;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 4px;
            background: #ffffff;
            border: 1px solid #f1f5f9;
        }

        .game-row:hover {
            border-color: #cbd5e1;
            background-color: #f8fafc;
        }

        .rank { font-weight: 700; color: #94a3b8; font-size: 0.9rem; }
        .name { font-weight: 700; color: #0f172a; font-size: 0.95rem; }
        .reason { color: #475569; font-size: 0.9rem; line-height: 1.4; }
        .genre { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; background: #f1f5f9; padding: 4px 8px; border-radius: 4px; text-align: center; }
        
        /* Match Bar */
        .match-container { background: #e2e8f0; height: 6px; border-radius: 10px; width: 100%; position: relative; }
        .match-bar { background: #6366f1; height: 6px; border-radius: 10px; }

        /* Sidebar Clean-up */
        [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
        .stSlider { margin-bottom: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

# 3. Backend Integration
@st.cache_resource
def load():
    if "SVD_URL" in st.secrets: os.environ["SVD_URL"] = st.secrets["SVD_URL"]
    return load_artifacts("artifacts")

@st.cache_data(show_spinner=False)
def cached_gemini_explain(api_key: str, recs_csv: str, seed_game: str = "", user_query: str = "") -> pd.DataFrame:
    rec_df = pd.read_csv(StringIO(recs_csv))
    return gemini_explain(api_key, rec_df, seed_game or None, user_query or None)

art = load()
all_games = sorted(art.df["game_name"].dropna().unique().tolist())
cluster_options = ["All Genres"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
desc_to_id = {v: k for k, v in art.cluster_labels.items()}

# 4. Sidebar Nav
with st.sidebar:
    st.markdown("### Search Parameters")
    mode = st.toggle("Switch to Natural Language Search", value=False)
    
    if not mode:
        game = st.selectbox("Based on this game:", all_games)
        user_query = ""
    else:
        user_query = st.text_area("Describe your ideal game:", placeholder="Example: Economic engine builder with high player interaction")
        game = ""

    st.divider()
    top_n = st.slider("Result Count", 5, 20, 8)
    sentiment_weight = st.slider("Vibe Weight", 0.0, 1.0, 0.25)
    selected_cluster = st.selectbox("Genre Filter", cluster_options)
    cluster_id = desc_to_id[selected_cluster] if selected_cluster != "All Genres" else None

# 5. Main Dashboard
st.markdown("""
    <div class="header-ui">
        <h2 style="margin:0; font-weight:800; letter-spacing:-1px;">MeepleMind <span style="color:#6366f1;">Pro</span></h2>
        <div style="font-size:0.85rem; color:#94a3b8; font-weight:500;">LIVE DATABASE INDEX v2.4</div>
    </div>
""", unsafe_allow_html=True)

run_query = (not mode and game) or (mode and user_query.strip())

if not run_query:
    st.info("Select a game or enter a query in the sidebar to generate recommendations.")
else:
    q_type = "game_name" if not mode else "text_query"
    q_val = game if not mode else user_query.strip()

    try:
        with st.spinner("Processing..."):
            recs = recommend(art, q_type, q_val, sentiment_weight, cluster_id, top_n)
            
            # AI reasoning
            if "GEMINI_API_KEY" in st.secrets:
                reasons_df = cached_gemini_explain(st.secrets["GEMINI_API_KEY"], recs.to_csv(index=False), game, user_query.strip())
            else:
                reasons_df = recs[["game_name"]].copy()
                reasons_df["reason"] = "Reasoning unavailable (No API Key)."

            display_df = recs.merge(reasons_df, on="game_name", how="left")

        # Table Header
        st.markdown("""
            <div style="display: grid; grid-template-columns: 50px 220px 1fr 180px 100px; gap: 20px; padding: 10px 16px; color: #94a3b8; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                <div>Rank</div>
                <div>Game Name</div>
                <div>AI Recommendation Logic</div>
                <div style="text-align:center;">Genre</div>
                <div>Match Score</div>
            </div>
        """, unsafe_allow_html=True)

        for i, row in enumerate(display_df.itertuples(index=False), start=1):
            # Normalize score for the bar (assuming scores are roughly 0-1)
            score_pct = min(max(row.score * 100, 10), 100)
            
            st.markdown(f"""
                <div class="game-row">
                    <div class="rank">#{i:02d}</div>
                    <div class="name">{html.escape(row.game_name)}</div>
                    <div class="reason">{html.escape(row.reason)}</div>
                    <div class="genre">{html.escape(str(row.cluster_label))}</div>
                    <div class="match-container">
                        <div class="match-bar" style="width: {score_pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Engine Error: {e}")
