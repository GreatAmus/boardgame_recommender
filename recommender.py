from dataclasses import dataclass
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from downloader import ensure_file

@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    X: np.ndarray
    embedder: object
    cluster_labels: dict

def load_artifacts(artifacts_dir="artifacts") -> RecommenderArtifacts:
    SVD_URL = "https://github.com/GreatAmus/boardgame_recommender/releases/tag/svd"
    ensure_file(SVD_URL, f"{artifacts_dir}/svd.joblib")
    
    df = pd.read_parquet(f"{artifacts_dir}/games.parquet")
    X = np.load(f"{artifacts_dir}/X.npy")

    tv   = joblib.load(f"{artifacts_dir}/tfidf.joblib")
    svd  = joblib.load(f"{artifacts_dir}/svd.joblib")
    norm = joblib.load(f"{artifacts_dir}/norm.joblib")

    embedder = make_pipeline(tv, svd, norm)

    cluster_labels = {
        0: "Tile-laying and spatial puzzles",
        1: "Air & naval wargames",
        2: "Conflict strategy and dudes-on-a-map",
        3: "Kids & family games",
        4: "Deduction & puzzles",
        5: "Word games",
        6: "Land wargames",
        7: "Dungeon crawler",
        8: "Light filler and abstract games",
        9: "Heavy euro and engine building",
        10: "Midweight tile and action games (mid Euros)",
        11: "Card games",
    }

    return RecommenderArtifacts(df=df, X=X, embedder=embedder, cluster_labels=cluster_labels)

def sims_from_game(df, X, game_name: str):
    idxs = df.index[df["game_name"] == game_name].tolist()
    if not idxs:
        raise ValueError(f"Game '{game_name}' not found.")
    idx = int(idxs[0])

    sims = cosine_similarity(X[idx:idx+1], X).ravel()
    sims[idx] = -1.0
    return sims, float(df.loc[idx, "sentiment"])

def sims_from_text(embedder, X, text: str):
    q = embedder.transform([text])
    return cosine_similarity(q, X).ravel()

def sentiment_match(s_game: float, s_target: float) -> float:
    # VADER compound range [-1, 1] => max diff 2
    return 1.0 - (abs(s_game - s_target) / 2.0)

def recommend(art: RecommenderArtifacts,
              query_type: str,
              query_value: str,
              sentiment_weight: float = 0.25,
              cluster_id: int | None = None,
              top_n: int = 10) -> pd.DataFrame:

    df, X = art.df, art.X

    if query_type == "game_name":
        sims, target_sent = sims_from_game(df, X, query_value)
    elif query_type == "text_query":
        sims = sims_from_text(art.embedder, X, query_value)
        target_sent = float(df["sentiment"].median())
    else:
        raise ValueError("query_type must be 'game_name' or 'text_query'.")

    candidates = np.arange(len(df))
    if cluster_id is not None:
        candidates = np.where(df["cluster"].to_numpy() == cluster_id)[0]

    sents = df["sentiment"].to_numpy(dtype=float)
    sent_match = np.array([sentiment_match(s, target_sent) for s in sents], dtype=float)

    combined = (1.0 - sentiment_weight) * sims + sentiment_weight * sent_match

    top_local = np.argsort(combined[candidates])[::-1][:top_n]
    top_idx = candidates[top_local]

    out = df.iloc[top_idx].copy()
    out["similarity"] = sims[top_idx]
    out["sentiment_match"] = sent_match[top_idx]
    out["combined_score"] = combined[top_idx]
    out["cluster_label"] = out["cluster"].map(art.cluster_labels).fillna("Unlabeled")

    return out.sort_values("combined_score", ascending=False).reset_index(drop=True)
