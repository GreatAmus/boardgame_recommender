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
        :root {
            --bg: #0b1020;
            --bg-accent: radial-gradient(circle at top left, rgba(99, 102, 241, 0.18), transparent 35%),
                         radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 30%),
                         linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
            --panel: rgba(15, 23, 42, 0.78);
            --panel-2: rgba(255, 255, 255, 0.04);
            --panel-solid: #111827;
            --border: rgba(255, 255, 255, 0.08);
            --border-strong: rgba(255, 255, 255, 0.14);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #818cf8;
            --accent-strong: #6366f1;
            --accent-soft: rgba(129, 140, 248, 0.14);
            --success-soft: rgba(16, 185, 129, 0.14);
            --shadow: 0 20px 50px rgba(0, 0, 0, 0.28);
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 12px;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--bg-accent);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        h1, h2, h3, h4, h5, h6, p, div, span, label {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .hero-card,
        .stat-card,
        .surface-card,
        .result-card,
        .empty-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }

        .hero-card {
            padding: 1.5rem;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            background: var(--panel-2);
            border: 1px solid var(--border);
            color: #c7d2fe;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 700;
            width: fit-content;
        }

        .hero-title {
            margin: 0.9rem 0 0.35rem 0;
            font-size: 2.25rem;
            line-height: 1.04;
            letter-spacing: -0.04em;
            font-weight: 800;
            color: #f8fafc;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.65;
            max-width: 54rem;
            margin: 0;
        }

        .hero-meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .stat-card {
            padding: 1rem 1.05rem;
            min-height: 90px;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .stat-value {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .layout-grid {
            display: grid;
            grid-template-columns: 360px minmax(0, 1fr);
            gap: 1rem;
            align-items: start;
        }

        .surface-card {
            padding: 1.1rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 800;
            color: #f8fafc;
            margin: 0 0 0.2rem 0;
            letter-spacing: -0.02em;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 1rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--accent-soft);
            color: #c7d2fe;
            border: 1px solid rgba(129, 140, 248, 0.24);
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .results-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .results-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .results-copy {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 0.25rem;
        }

        .result-card {
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.85rem;
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }

        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(129, 140, 248, 0.32);
            background: rgba(15, 23, 42, 0.9);
        }

        .result-row {
            display: grid;
            grid-template-columns: 44px minmax(0, 1fr);
            gap: 0.9rem;
            align-items: start;
        }

        .rank-badge {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(16, 185, 129, 0.18));
            border: 1px solid rgba(129, 140, 248, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #eef2ff;
            font-weight: 800;
            font-size: 0.95rem;
        }

        .game-title {
            color: #f8fafc;
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.3;
            margin: 0 0 0.4rem 0;
        }

        .game-reason {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.65;
            margin: 0;
            white-space: normal;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        .empty-card {
            padding: 1.2rem;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.6;
        }

        .hint-list {
            margin: 0.2rem 0 0 1rem;
            padding: 0;
            color: var(--muted);
        }

        .hint-list li {
            margin-bottom: 0.35rem;
        }

        div[data-testid="stRadio"] > div {
            gap: 0.5rem;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.35rem;
        }

        div[data-testid="stRadio"] label {
            min-height: 46px;
            border-radius: 12px !important;
            justify-content: center;
            background: transparent !important;
            border: none !important;
            color: var(--muted) !important;
            font-weight: 700 !important;
            margin: 0 !important;
        }

        div[data-testid="stRadio"] label:has(input:checked) {
            background: rgba(129, 140, 248, 0.14) !important;
            color: #eef2ff !important;
            box-shadow: inset 0 0 0 1px rgba(129, 140, 248, 0.18);
        }

        .stSelectbox label,
        .stTextArea label,
        .stSlider label {
            color: #dbe4f0 !important;
            font-size: 0.9rem !important;
            font-weight: 700 !important;
        }

        .stTextArea textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div {
            border-radius: 14px !important;
        }

        .stTextArea textarea,
        .stSelectbox > div > div > div,
        .stSlider,
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.03) !important;
            color: #f8fafc !important;
        }

        .stSelectbox div[data-baseweb="select"] > div,
        .stTextArea textarea {
            border: 1px solid var(--border-strong) !important;
        }

        .stTextArea textarea::placeholder {
            color: #7c8aa0 !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        .stTextArea textarea,
        .stSlider label,
        .stMarkdown,
        .stCaption {
            color: #f8fafc;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: var(--accent-strong) !important;
        }

        .stSlider [role="slider"] {
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.18) !important;
        }

        .stAlert {
            border-radius: 16px;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.6rem;
            line-height: 1.5;
        }

        @media (max-width: 1100px) {
            .hero,
            .layout-grid {
                grid-template-columns: 1fr;
            }

            .hero-meta {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero-card,
            .surface-card,
            .result-card,
            .empty-card,
            .stat-card {
                border-radius: 18px;
            }

            .hero-title {
                font-size: 1.8rem;
            }

            .results-toolbar {
                flex-direction: column;
                align-items: start;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_result_card(index: int, game_name: str, reason: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-row">
                <div class="rank-badge">{index}</div>
                <div>
                    <div class="game-title">{html.escape(str(game_name))}</div>
                    <p class="game-reason">{html.escape(str(reason))}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-card">
            <div>
                <div class="hero-badge">🎲 Smart discovery for tabletop players</div>
                <h1 class="hero-title">Board Game Recommender</h1>
                <p class="hero-subtitle">
                    Search by a game you already love or describe the experience you want.
                    The app returns the best matching games from review-driven recommendation artifacts.
                </p>
            </div>
        </div>
        <div class="hero-meta">
            <div class="stat-card">
                <div class="stat-label">Games indexed</div>
                <div class="stat-value">{len(all_games):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Search modes</div>
                <div class="stat-value">2</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Clusters</div>
                <div class="stat-value">{len(cluster_options) - 1}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="layout-grid">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search controls</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Choose how you want to search, then refine the ranking with a few lightweight filters.</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Search mode",
        ["Game name", "Natural language query"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Game name":
        game = st.selectbox(
            "Pick a game",
            all_games,
            help="Use a known game as the seed for similar recommendations.",
        )
        user_query = ""
    else:
        user_query = st.text_area(
            "Describe what you want",
            placeholder="Strategic engine-building game with strong replayability and low direct conflict",
            height=120,
            help="Describe mechanics, mood, complexity, player interaction, or replayability.",
        )
        game = ""

    st.markdown('<div style="height: 0.35rem"></div>', unsafe_allow_html=True)

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
    )

    sentiment_weight = st.slider(
        "Sentiment weight",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Higher values give more influence to review sentiment in the ranking.",
    )

    selected_cluster_desc = st.selectbox(
        "Cluster",
        cluster_options,
        help="Limit recommendations to a specific game cluster, or search across all clusters.",
    )

    cluster_id = None
    if selected_cluster_desc != "All clusters":
        cluster_id = desc_to_id[selected_cluster_desc]

    st.markdown(
        '<div class="footer-note">Tip: natural-language search works best when you mention mechanics, complexity, pacing, or interaction style.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

run_query = False
if mode == "Game name" and game:
    run_query = True
if mode == "Natural language query" and user_query.strip():
    run_query = True

with st.container():
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)

    if mode == "Game name":
        context_text = f'Based on “{game}”' if game else "Choose a game to begin"
        subtitle = "Similar games ranked using the same recommendation pipeline and optional explanation generation."
    else:
        context_text = f'Query: “{user_query.strip()}”' if user_query.strip() else "Describe the kind of game you want"
        subtitle = "Games that best match your natural-language request, filtered and ranked by your settings."

    st.markdown(
        f"""
        <div class="results-toolbar">
            <div>
                <div class="results-title">Recommendations</div>
                <div class="results-copy">{html.escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="pill">✨ {html.escape(context_text)}</div>',
        unsafe_allow_html=True,
    )

    if not run_query:
        st.markdown(
            """
            <div class="empty-card">
                <strong style="color:#f8fafc; display:block; margin-bottom:0.45rem;">Start with a search</strong>
                Recommendations will appear here once you either choose a seed game or enter a natural-language query.
                <ul class="hint-list">
                    <li>Use <strong>Game name</strong> when you want titles similar to a favorite game.</li>
                    <li>Use <strong>Natural language query</strong> when you want to describe mechanics or play feel.</li>
                    <li>Adjust cluster and sentiment weight to fine-tune the ranking.</li>
                </ul>
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
                with st.spinner("Writing concise recommendation reasons..."):
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

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
