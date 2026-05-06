from __future__ import annotations

from pathlib import Path

import pandas as pd

from cinematch.joint import score_columns
from cinematch.models import TrainedUserModel, load_artifacts


def score_candidates(
    candidates: pd.DataFrame,
    users: list[str] | None = None,
    model_dir: Path = Path("models"),
    lam: float = 1.0,
) -> pd.DataFrame:
    encoder, models = load_artifacts(model_dir)
    selected_users = users or sorted(models)
    missing = [user for user in selected_users if user not in models]
    if missing:
        raise ValueError(f"No trained model for users: {missing}")

    X = encoder.transform(candidates)
    out = candidates[["title", "year"]].copy()
    prediction_cols = []
    for user in selected_users:
        col = f"{user}_pred"
        prediction_cols.append(col)
        out[col] = models[user].predict(X)

    means, stds, joint = score_columns(out[prediction_cols].to_numpy(), lam=lam)
    out["mean_prediction"] = means
    out["std_prediction"] = stds
    out["group_score"] = joint
    return out.sort_values("group_score", ascending=False).reset_index(drop=True)


def recommend_from_csv(
    candidates_csv: Path,
    users: list[str] | None = None,
    model_dir: Path = Path("models"),
    top_n: int = 10,
    lam: float = 1.0,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_csv)
    scored = score_candidates(candidates, users=users, model_dir=model_dir, lam=lam)
    return scored.head(top_n)
