import os
from io import StringIO
import html
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

# Configuration
st.set_page_config(
    page_title="MeepleMind | Board Game Discovery",
    page_icon="🎲",
    layout="wide",
)

# --- CUSTOM CSS (The "Glow Up") ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #fcfcfd;
        }

        /* Hero Section */
        .hero-container {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            padding: 3rem 2rem;
            border-radius: 24px;
            color: white;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        /* Glassmorphism Sidebar/Panel */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        .stSelectbox, .stSlider, .stTextArea {
            margin-bottom: 1.5rem;
        }

        /* Recommendation Cards */
        .rec-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.2rem;
            transition: all 0.3s ease;
        }

        .rec-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 20px -8px rgba(0,0,0,0.1);
            border-color: #6366f1;
        }

        .rec-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .rank-badge {
            background: #6366f1;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
        }

        .game-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e293b;
        }

        .reason-text {
            color: #64748b;
            line-height: 1.6;
            font-size: 0.95rem;
            border-left: 3px solid #e2e8f0;
            padding-left: 1rem;
        }

        .chip {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #f1f5f9;
            color: #475569;
            margin-top: 10px;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
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
all_games = sorted(art.df["game_name"].dropna().unique().tolist())
cluster_options = ["All Genres"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
desc_to_id = {v: k for k, v in art.cluster_labels.items()}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Filters")
    
    mode = st.radio(
        "Search Method",
        ["Similar to a Game", "Natural Language"],
    )

    if mode == "Similar to a Game":
        game = st.selectbox("Select a Game", all_games)
        user_query = ""
    else:
        user_query = st.text_area("What are you in the mood for?", 
                                placeholder="A space-themed engine builder with dice rolling...")
        game = ""

    st.divider()
    
    with st.expander("Fine-Tuning", expanded=True):
        top_n = st.slider("Max Results", 3, 15, 6)
        sentiment_weight = st.slider("Vibe Match (Sentiment)", 0.0, 1.0, 0.25)
        selected_cluster = st.selectbox("Filter by Genre", cluster_options)

    cluster_id = desc_to_id[selected_cluster] if selected_cluster != "All Genres" else None

# --- MAIN CONTENT ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">MeepleMind</div>
        <div style="opacity: 0.9; font-size: 1.1rem;">The intelligent way to find your next favorite board game.</div>
    </div>
""", unsafe_allow_html=True)

# Logic to run query
run_query = (mode == "Similar to a Game" and game) or (mode == "Natural Language" and user_query.strip())

if not run_query:
    st.info("👈 Use the sidebar to select a game or describe your ideal playstyle!")
else:
    query_type = "game_name" if mode == "Similar to a Game" else "text_query"
    query_value = game if mode == "Similar to a Game" else user_query.strip()

    try:
        with st.spinner("Analyzing mechanics and sentiments..."):
            recs = recommend(
                art=art,
                query_type=query_type,
                query_value=query_value,
                sentiment_weight=sentiment_weight,
                cluster_id=cluster_id,
                top_n=top_n,
            )

        # Gemini Logic
        if "GEMINI_API_KEY" in st.secrets:
            with st.spinner("Consulting the AI Oracle..."):
                reasons_df = cached_gemini_explain(
                    api_key=st.secrets["GEMINI_API_KEY"],
                    recs_csv=recs.to_csv(index=False),
                    seed_game=game,
                    user_query=user_query.strip(),
                )
        else:
            reasons_df = recs[["game_name"]].copy()
            reasons_df["reason"] = "Add a Gemini API key to see AI-powered reasoning."

        display_df = recs[["game_name", "cluster_label"]].merge(reasons_df, on="game_name", how="left")
        
        # Grid Layout for Results
        col1, col2 = st.columns(2)
        
        for i, row in enumerate(display_df.itertuples(index=False), start=1):
            target_col = col1 if i % 2 != 0 else col2
            
            with target_col:
                st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-header">
                            <div class="rank-badge">{i}</div>
                            <div class="game-title">{html.escape(row.game_name)}</div>
                        </div>
                        <div class="reason-text">
                            {html.escape(row.reason or "No specific reasoning available.")}
                        </div>
                        <div class="chip">{html.escape(str(row.cluster_label))}</div>
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Footer
st.markdown("---")
st.caption("Data powered by SVD Embeddings & Google Gemini. Handcrafted for Board Game Enthusiasts.")
