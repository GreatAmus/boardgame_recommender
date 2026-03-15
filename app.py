import os
from io import StringIO
import html
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend, gemini_explain

# 1. Page Config
st.set_page_config(
    page_title="Board Game Index",
    page_icon="📋",
    layout="wide",
)

# 2. Optimized High-Density CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1e293b;
        }

        .block-container {
            padding: 1rem 3rem !important;
            max-width: 1400px;
        }

        /* Top Header */
        .top-bar {
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }

        .top-bar h1 {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
        }

        /* The Editorial Table Style */
        .rec-row {
            display: grid;
            grid-template-columns: 40px 220px 1fr 180px;
            gap: 15px;
            padding: 12px 15px;
            border-bottom: 1px solid #f1f5f9;
            align-items: center;
            background: white;
        }

        .rec-row:hover {
            background-color: #f8fafc;
        }

        .rec-rank {
            font-weight: 700;
            color: #94a3b8;
            font-size: 0.9rem;
        }

        .rec-name {
            font-weight: 700;
            font-size: 0.95rem;
            color: #0f172a;
        }

        .rec-reason {
            font-size: 0.9rem;
            color: #475569;
            line-height: 1.5;
        }

        .rec-tag {
            font-size: 0.7rem;
            font-weight: 700;
            background: #f1f5f9;
            padding: 4px 8px;
            border-radius: 4px;
            text-align: center;
            text-transform: uppercase;
            color: #64748b;
        }

        /* Sidebar & Button Styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0;
        }
        
        div.stButton > button {
            width: 100%;
            background-color: #6366f1;
            color: white;
            border: none;
            padding: 0.5rem;
            font-weight: 600;
            border-radius: 6px;
        }
        
        div.stButton > button:hover {
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
    return gemini_explain(api_key, rec_df, seed_game or None, user_query or None)

art = load()
all_games = sorted(art.df["game_name"].dropna().unique().tolist())
cluster_options = ["All Genres"] + [art.cluster_labels[k] for k in sorted(art.cluster_labels)]
desc_to_id = {v: k for k, v in art.cluster_labels.items()}

# 4. Sidebar Nav with Form (Search Button)
with st.sidebar:
    st.subheader("Search Engine")
    
    # Toggle switch outside the form so it resets the fields appropriately
    mode = st.toggle("Natural Language Mode", value=False)
    
    with st.form("search_form"):
        if not mode:
            game_input = st.selectbox("Based on this game:", all_games)
            query_input = ""
        else:
            query_input = st.text_area("Describe the gameplay:", placeholder="e.g. Cooperative dungeon crawler with deck building")
            game_input = ""

        st.divider()
        top_n = st.slider("Results", 5, 20, 10)
        sentiment_weight = st.slider("Vibe Weight", 0.0, 1.0, 0.2)
        selected_cluster = st.selectbox("Genre Filter", cluster_options)
        
        # The Manual Search Button
        submit_button = st.form_submit_button("Find Games")

cluster_id = desc_to_id[selected_cluster] if selected_cluster != "All Genres" else None

# 5. Main Layout
st.markdown("""
    <div class="top-bar">
        <h1>Board Game Index</h1>
        <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">SEARCH & DISCOVERY TERMINAL</div>
    </div>
""", unsafe_allow_html=True)

# Run query ONLY when button is clicked
if not submit_button:
    st.info("👈 Configure your search in the sidebar and click 'Find Games' to begin.")
else:
    q_type = "game_name" if not mode else "text_query"
    q_val = game_input if not mode else query_input.strip()

    if q_type == "text_query" and not q_val:
        st.warning("Please enter a description before searching.")
    else:
        try:
            with st.spinner("Fetching matches..."):
                recs = recommend(art, q_type, q_val, sentiment_weight, cluster_id, top_n)

            if "GEMINI_API_KEY" in st.secrets:
                reasons_df = cached_gemini_explain(
                    st.secrets["GEMINI_API_KEY"], recs.to_csv(index=False), game_input, query_input.strip()
                )
            else:
                reasons_df = recs[["game_name"]].copy()
                reasons_df["reason"] = "Add GEMINI_API_KEY for AI analysis."

            display_df = recs.merge(reasons_df, on="game_name", how="left")

            # Table Header
            st.markdown("""
                <div style="display: grid; grid-template-columns: 40px 220px 1fr 180px; gap: 15px; padding: 8px 15px; background: #0f172a; color: white; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
                    <div>#</div>
                    <div>Game Title</div>
                    <div>Recommendation Logic</div>
                    <div style="text-align: center;">Genre</div>
                </div>
            """, unsafe_allow_html=True)

            # Rows
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
            st.error(f"Search failed: {e}")
