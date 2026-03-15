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

# The NEW Google GenAI SDK
from pydantic import BaseModel
from google import genai
from google.genai import types

@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    X: np.ndarray
    embedder: object
    cluster_labels: dict

# Define the schema for structured output
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
        raise RuntimeError(f"Failed to download from {url}: {e}")

def load_artifacts(artifacts_dir: str = "artifacts") -> RecommenderArtifacts:
    df = pd.read_parquet(f"{artifacts_dir}/games.parquet")
    X = np.load(f"{artifacts_dir}/X.npy")
    
    # Load model pipeline components
    tv = joblib.load(f"{artifacts_dir}/tfidf.joblib")
    svd = joblib.load(f"{artifacts_dir}/svd.joblib")
    norm = joblib.load(f"{artifacts_dir}/norm.joblib")
    embedder = make_pipeline(tv, svd, norm)

    cluster_labels = {
        0: "Tile-laying and spatial puzzles", 1: "Air & naval wargames",
        2: "Conflict strategy", 3: "Kids & family games",
        4: "Deduction & puzzles", 5: "Word games",
        6: "Land wargames", 7: "Dungeon crawler",
        8: "Light filler", 9: "Heavy euro",
        10: "Midweight euro", 11: "Card games",
    }
    return RecommenderArtifacts(df=df, X=X, embedder=embedder, cluster_labels=cluster_labels)

def sentiment_match(s_game: float, s_target: float) -> float:
    return 1.0 - (abs(float(s_game) - float(s_target)) / 2.0)

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
        idxs = df.index[df["game_name"] == query_value].tolist()
        if not idxs: raise ValueError(f"Game '{query_value}' not found.")
        idx = int(idxs[0])
        sims = cosine_similarity(X[idx:idx + 1], X).ravel()
        sims[idx] = -1.0
        target_sent = float(df.loc[idx, "sentiment"]) if "sentiment" in df.columns else 0.0
    else:
        q = art.embedder.transform([query_value])
        sims = cosine_similarity(q, X).ravel()
        target_sent = float(df["sentiment"].median()) if "sentiment" in df.columns else 0.0

    candidates = np.arange(len(df))
    if cluster_id is not None:
        candidates = np.where(df["cluster"].to_numpy() == cluster_id)[0]

    if "sentiment" in df.columns:
        sents = df["sentiment"].to_numpy(dtype=float)
        sent_scores = np.array([sentiment_match(s, target_sent) for s in sents])
        final_scores = (1.0 - sentiment_weight) * sims + sentiment_weight * sent_scores
    else:
        sent_scores = np.zeros(len(df))
        final_scores = sims

    ranked = candidates[np.argsort(final_scores[candidates])[::-1][:top_n]]
    out = df.iloc[ranked].copy()
    out["score"] = final_scores[ranked]
    return out.reset_index(drop=True)

def gemini_explain(
    api_key: str,
    rec_df: pd.DataFrame,
    seed_game: Optional[str] = None,
    user_query: Optional[str] = None,
) -> pd.DataFrame:
    # Use the new Client syntax
    client = genai.Client(api_key=api_key)
    context = seed_game or user_query or "the user's preference"

    prompt = (
        f"Explain why these board games match this context: {context}. "
        "Keep reasons concise (1-2 sentences) and focus on mechanics or theme."
    )
    
    game_list = "\n".join(f"- {name}" for name in rec_df["game_name"])

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{prompt}\n\nGames:\n{game_list}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RecommendationReasons,
            temperature=0.2,
        ),
    )

    # The new SDK parses directly into the schema if specified
    parsed = response.parsed
    if not parsed:
        return pd.DataFrame([{"game_name": n, "reason": "Recommended based on search."} for n in rec_df["game_name"]])

    out_df = pd.DataFrame([{"game_name": r.game_name, "reason": r.reason} for r in parsed.recommendations])
    
    # Merge back to ensure we keep the original recommendation order
    final_df = rec_df[["game_name"]].merge(out_df, on="game_name", how="left")
    final_df["reason"] = final_df["reason"].fillna("Matches your mechanical preferences.")
    return final_df
