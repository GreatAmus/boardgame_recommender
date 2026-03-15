from dataclasses import dataclass
import json
import os
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline


@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    X: np.ndarray
    embedder: object
    cluster_labels: dict


def ensure_file(url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    try:
        urllib.request.urlretrieve(url, local_path)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download file. HTTP status: {e.code}. URL: {url}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}. Error: {e}") from e

    with open(local_path, "rb") as f:
        first_bytes = f.read(200)

    if first_bytes.startswith(b"<!DOCTYPE html") or first_bytes.startswith(b"<html") or b"<html" in first_bytes.lower():
        raise RuntimeError(
            f"Downloaded file at {local_path} looks like HTML, not a model binary. "
            "Check that the URL is the direct asset download link."
        )


def load_artifacts(artifacts_dir: str = "artifacts") -> RecommenderArtifacts:
    df = pd.read_parquet(f"{artifacts_dir}/games.parquet")
    X = np.load(f"{artifacts_dir}/X.npy")

    tfidf_path = f"{artifacts_dir}/tfidf.joblib"
    svd_path = f"{artifacts_dir}/svd.joblib"
    norm_path = f"{artifacts_dir}/norm.joblib"

    tfidf_url = os.environ.get("TFIDF_URL", "")
    svd_url = os.environ.get("SVD_URL", "")
    norm_url = os.environ.get("NORM_URL", "")

    if not os.path.exists(tfidf_path):
        if not tfidf_url:
            raise FileNotFoundError("Missing artifacts/tfidf.joblib and TFIDF_URL not set.")
        ensure_file(tfidf_url, tfidf_path)

    if not os.path.exists(svd_path):
        if not svd_url:
            raise FileNotFoundError("Missing artifacts/svd.joblib and SVD_URL not set.")
        ensure_file(svd_url, svd_path)

    if not os.path.exists(norm_path):
        if not norm_url:
            raise FileNotFoundError("Missing artifacts/norm.joblib and NORM_URL not set.")
        ensure_file(norm_url, norm_path)

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
    out["similarity"] = sims[ranked]
    out["sentiment_match"] = sent_scores[ranked]

    if "cluster" in out.columns:
        out["cluster_label"] = out["cluster"].map(art.cluster_labels).fillna("Unlabeled")

    return out.reset_index(drop=True)


def format_recs_for_prompt(
    rec_df: pd.DataFrame,
    seed_game: str | None = None,
    user_query: str | None = None,
) -> str:
    lines: list[str] = []

    if seed_game:
        lines.append(f"Seed game: {seed_game}")
    if user_query:
        lines.append(f"User query: {user_query}")

    lines.append("Recommendations:")

    for r in rec_df.itertuples(index=False):
        score = getattr(r, "score", None)
        score_s = f"{score:.4f}" if isinstance(score, (int, float, np.floating)) else str(score)

        cluster_label = getattr(r, "cluster_label", getattr(r, "cluster", None))
        sentiment = getattr(r, "sentiment", None)
        sent_s = f"{sentiment:.3f}" if isinstance(sentiment, (int, float, np.floating)) else str(sentiment)

        lines.append(
            f"- {r.game_name} (cluster={cluster_label}, sentiment={sent_s}, score={score_s})"
        )

    return "\n".join(lines)


def gemini_explain(
    api_key: str,
    rec_df: pd.DataFrame,
    seed_game: str | None = None,
    user_query: str | None = None,
    model: str = "gemini-2.0-flash",
) -> pd.DataFrame:
    if seed_game:
        task_context = "Explain why each recommendation is a good match for the selected seed game."
    else:
        task_context = "Explain why each recommendation is a good match for the user's natural language query."

    prompt = (
        "You are helping explain board game recommendations.\n"
        f"{task_context}\n"
        "For each recommended game, give 1 concise reason in plain English.\n"
        "Return ONLY valid JSON in this exact format:\n"
        '[{"game_name": "name here", "reason": "reason here"}]\n'
        "Do not include markdown fences. Do not include any text before or after the JSON.\n\n"
        + format_recs_for_prompt(rec_df=rec_df, seed_game=seed_game, user_query=user_query)
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        fallback = rec_df[["game_name"]].copy()
        fallback["reason"] = f"Could not generate explanation: {e}"
        return fallback

    try:
        parsed = json.loads(text)
        out = pd.DataFrame(parsed)
        merged = rec_df[["game_name"]].merge(out, on="game_name", how="left")
        merged["reason"] = merged["reason"].fillna("No explanation returned.")
        return merged[["game_name", "reason"]]
    except Exception:
        fallback = rec_df[["game_name"]].copy()
        fallback["reason"] = text
        return fallback
