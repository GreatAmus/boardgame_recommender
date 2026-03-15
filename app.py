import os
from io import StringIO
import html
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

# 1. Page Config
st.set_page_config(
    page_title="MeepleMind | Night Mode",
    page_icon="♟️",
    layout="wide",
)

# 2. Immersive CSS
st.markdown("""
    <style>
        /* Modern Dark Theme Overrides */
        .stApp {
            background-color: #0f172a;
            color: #f1f5f9;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1e293b !important;
            border-right: 1px solid #334155;
        }

        /* The Hero Area */
        .hero {
            padding: 2rem;
            border-radius: 20px;
            background: radial-gradient(circle at top right, #334155, #0f172a);
            border: 1px solid #334155;
            margin-bottom: 2rem;
            text-align: left;
        }

        .hero h1 {
            color: #818cf8;
            font-weight: 800;
            margin-bottom: 0px;
        }

        /* Recommendation Cards */
        .game-card {
            background: #1e293b;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 1.5rem;
            border-bottom: 4px solid #4f46e5; /* Neon accent */
            transition: transform 0.2s ease;
        }

        .game-card:hover {
            transform: scale(1.02);
            background: #334155;
        }

        .rank-num {
            font-family: 'Monospace';
            color: #818cf8;
            font-size: 0.8rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .title-text {
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 12px;
        }

        .reason-box {
            background: rgba(15, 23, 42, 0.5);
            padding: 12px;
            border-radius: 8px;
            font-style: italic;
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* Tags */
        .genre-tag {
            display: inline-block;
            margin-top: 15px;
            padding: 4px 10px;
            background: #4f46e5;
            color: white;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
            text-transform: uppercase;
        }

        /* Customizing Streamlit Widgets */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #4f46e5;
            color: white;
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

# 4. Sidebar Nav
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=80)
    st.title("Search Terminal")
    
    mode = st.segmented_control(
        "Input Mode", ["Classic", "Vibe Search"], default="Classic"
    )

    if mode == "Classic":
        game = st.selectbox("Seed Game", all_games)
        user_query =
