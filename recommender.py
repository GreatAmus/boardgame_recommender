from dataclasses import dataclass
import json
import os
import urllib.request
import urllib.error
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

# Gemini Imports
import google.generativeai as genai
from pydantic import BaseModel

@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    X: np.ndarray
    embedder: object
    cluster_labels: dict

# Schema for Gemini Structured Output
class RecommendationReason(BaseModel):
    game_name: str
    reason: str

class RecommendationReasons(BaseModel):
    recommendations: List[RecommendationReason]

def ensure_file(url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}. Error: {e}")

def load_artifacts(artifacts_dir: str = "artifacts") -> RecommenderArtifacts:
    # Ensure directory exists
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    df = pd.read_parquet(f"{artifacts_dir}/games.parquet")
    X = np.load(f"{artifacts_dir}/X.npy")

    tfidf_path = f"{artifacts_dir}/tfidf.joblib"
    svd_path = f"{artifacts_dir}/svd.joblib"
    norm_path = f"{artifacts_dir}/norm.joblib"

    # SVD is usually the large file stored externally
    svd_url = os.environ.get("SVD_URL", "")
    if svd_url:
        ensure_file(svd_url, svd_path)

    tv = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)
    norm = joblib.load(norm_path)
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

def sentiment_match(s_game: float, s_target: float) -> float:
    return 1.0 - (abs(float(s_game) - float(s_target)) / 2.0)

def sims_from_game(df: pd.DataFrame, X: np.ndarray, game_name: str) -> tuple[np.ndarray, float]:
    idxs = df.index[df["game_name"] == game_name].tolist()
    if not idxs:
        raise ValueError(f"Game '{game_name}' not found.")
    idx = int(idxs[0])
    sims = cosine_similarity(X[idx:idx + 1], X).ravel()
    sims[idx] = -1.0
    target_sent = float(df.loc[idx, "sentiment"]) if "sentiment" in df.columns else 0.0
    return sims, target_sent

def sims_from_text(embedder, X: np.ndarray, text: str, df: pd.DataFrame) -> tuple[np.ndarray, float]:
    text = text.strip()
    if not text:
        raise ValueError("Please enter a natural language query.")
    q = embedder.transform([text])
    sims = cosine_similarity(q, X).ravel()
    target_sent = float(df["sentiment"].median()) if "sentiment" in df.columns else 0.0
    return sims, target_sent

def recommend(
    art: RecommenderArtifacts,
    query_type: str,
    query_value: str,
    sentiment_weight: float = 0.25,
    cluster_id: int | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    df = art.df
    X = art.X

    if query_type == "game_name":
        sims, target_sent = sims_from_game(df, X, query_value)
    elif query_type == "text_query":
        sims, target_sent = sims_from_text(art.embedder, X, query_value, df)
    else:
        raise ValueError("query_type must be 'game_name' or 'text_query'.")

    candidates = np.arange(len(df))
    if cluster_id is not None and "cluster" in df.columns:
        candidates = np.where(df["cluster"].to_numpy() == cluster_id)[0]

    if "sentiment" in df.columns:
        sents = df["sentiment"].to_numpy(dtype=float)
        sent_scores = np.array([sentiment_match(s, target_sent) for s in sents], dtype=float)
        final_scores = (1.0 - sentiment_weight) * sims + sentiment_weight * sent_scores
    else:
        sent_scores = np.zeros(len(df), dtype=float)
        final_scores = sims

    ranked = candidates[np.argsort(final_scores[candidates])[::-1][:top_n]]
    out = df.iloc[ranked].copy()
    out["score"] = final_scores[ranked]
    
    if "cluster" in out.columns:
        out["cluster_label"] = out["cluster"].map(art.cluster_labels).fillna("Unlabeled")

    return out.reset_index(drop=True)

def gemini_explain(
    api_key: str,
    rec_df: pd.DataFrame,
    seed_game: Optional[str] = None,
    user_query: Optional[str] = None,
) -> pd.DataFrame:
    # Setup Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash") # Or gemini-2.0-flash-exp

    context = seed_game or user_query or "the user's request"

    prompt = (
        "You are an expert board game recommender. Given a user's context and a list of games, "
        "explain why each game fits the context.\n"
        f"Context: {context}\n\n"
        "Return the reasons as a structured JSON object containing a list of objects "
        "with 'game_name' and 'reason' keys."
    )
    
    game_list = "\n".join(f"- {name}" for name in rec_df["game_name"].tolist())
    full_prompt = f"{prompt}\n\nGames to explain:\n{game_list}"

    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=RecommendationReasons,
        ),
    )

    try:
        raw_json = json.loads(response.text)
        out_df = pd.DataFrame(raw_json["recommendations"])
    except Exception:
        # Fallback if parsing fails
        out_df = pd.DataFrame([{"game_name": n, "reason": "Highly rated and similar theme."} for n in rec_df["game_name"]])

    final_df = rec_df[["game_name"]].merge(out_df, on="game_name", how="left")
    final_df["reason"] = final_df["reason"].fillna("Recommended based on similarity.")
    return final_df
