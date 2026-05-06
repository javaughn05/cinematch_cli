from __future__ import annotations

from pathlib import Path

import pandas as pd

from cinematch.joint import joint_score
from cinematch.models import load_artifacts


def load_movie_pool(
    enriched_dir: Path = Path("data/enriched"),
    extra_csv: Path | None = None,
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(Path(enriched_dir).glob("*.csv"))]
    if extra_csv is not None:
        frames.append(pd.read_csv(extra_csv))
    if not frames:
        raise ValueError("No movie metadata available to score")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["title", "year"])


def score_movie(
    title: str,
    year: int | None = None,
    users: list[str] | None = None,
    model_dir: Path = Path("models"),
    enriched_dir: Path = Path("data/enriched"),
    extra_csv: Path | None = None,
    lam: float = 1.0,
) -> dict:
    encoder, models = load_artifacts(model_dir)
    selected_users = users or sorted(models)
    missing = [user for user in selected_users if user not in models]
    if missing:
        raise ValueError(f"No trained model for users: {missing}")

    pool = load_movie_pool(enriched_dir, extra_csv)
    matches = pool[pool["title"].str.casefold() == title.casefold()]
    if year is not None:
        matches = matches[matches["year"].astype("Int64") == int(year)]
    if matches.empty:
        raise ValueError(
            f"No metadata found for {title!r}. Enrich it first or pass it in extra_csv."
        )

    movie = matches.iloc[[0]]
    X = encoder.transform(movie)
    predictions = {
        user: float(models[user].predict(X)[0])
        for user in selected_users
    }
    return {
        "title": str(movie.iloc[0]["title"]),
        "year": None if pd.isna(movie.iloc[0]["year"]) else int(movie.iloc[0]["year"]),
        "predictions": predictions,
        "joint_score": joint_score(list(predictions.values()), lam=lam),
    }
