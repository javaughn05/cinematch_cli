from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED_REQUIRED_COLUMNS = {
    "title",
    "year",
    "rating",
    "genres",
    "director",
    "top_cast",
    "runtime",
    "original_language",
    "community_avg",
}


def split_multi(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def decade_from_year(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return f"{year // 10 * 10}s"


def load_enriched_user_frames(enriched_dir: Path = Path("data/enriched")) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(enriched_dir).glob("*.csv")):
        df = pd.read_csv(path)
        missing = ENRICHED_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        df = df[df["rating"].notna()].copy()
        if not df.empty:
            frames[path.stem] = df
    if not frames:
        raise ValueError(f"No enriched user CSVs found in {enriched_dir}")
    return frames


@dataclass
class FeatureEncoder:
    """Shared movie metadata encoder for user data and candidate pools."""

    top_directors: int = 50
    top_cast: int = 100
    genres: list[str] = field(default_factory=list)
    directors: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    decades: list[str] = field(default_factory=list)

    def fit(self, frames: dict[str, pd.DataFrame] | list[pd.DataFrame]) -> "FeatureEncoder":
        dfs = list(frames.values()) if isinstance(frames, dict) else list(frames)
        genre_counter: Counter[str] = Counter()
        director_counter: Counter[str] = Counter()
        cast_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()
        decade_counter: Counter[str] = Counter()

        for df in dfs:
            for _, row in df.iterrows():
                genre_counter.update(split_multi(row.get("genres")))
                director_counter.update(split_multi(row.get("director")))
                cast_counter.update(split_multi(row.get("top_cast")))
                lang = row.get("original_language")
                if pd.notna(lang) and str(lang).strip():
                    language_counter[str(lang).strip()] += 1
                decade = decade_from_year(row.get("year"))
                if decade:
                    decade_counter[decade] += 1

        self.genres = sorted(genre_counter)
        self.directors = [name for name, _ in director_counter.most_common(self.top_directors)]
        self.cast = [name for name, _ in cast_counter.most_common(self.top_cast)]
        self.languages = sorted(language_counter)
        self.decades = sorted(decade_counter)
        return self

    @property
    def feature_names(self) -> list[str]:
        names = [f"genre:{x}" for x in self.genres]
        names += [f"director:{x}" for x in self.directors]
        names += ["director:other"]
        names += [f"cast:{x}" for x in self.cast]
        names += ["cast:other"]
        names += [f"language:{x}" for x in self.languages]
        names += [f"decade:{x}" for x in self.decades]
        names += ["runtime_norm", "community_avg_norm"]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = [self._encode_row(row) for _, row in df.iterrows()]
        if not rows:
            return np.empty((0, len(self.feature_names)))
        return np.array(rows, dtype=float)

    def transform_user(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rated = df[df["rating"].notna()].copy()
        return self.transform(rated), rated["rating"].astype(float).to_numpy()

    def _encode_row(self, row: pd.Series) -> list[float]:
        values: list[float] = []

        row_genres = set(split_multi(row.get("genres")))
        values.extend(1.0 if item in row_genres else 0.0 for item in self.genres)

        row_directors = set(split_multi(row.get("director")))
        known_directors = set(self.directors)
        values.extend(1.0 if item in row_directors else 0.0 for item in self.directors)
        values.append(1.0 if row_directors - known_directors else 0.0)

        row_cast = set(split_multi(row.get("top_cast")))
        known_cast = set(self.cast)
        values.extend(1.0 if item in row_cast else 0.0 for item in self.cast)
        values.append(1.0 if row_cast - known_cast else 0.0)

        language = row.get("original_language")
        language = str(language).strip() if pd.notna(language) else ""
        values.extend(1.0 if item == language else 0.0 for item in self.languages)

        decade = decade_from_year(row.get("year"))
        values.extend(1.0 if item == decade else 0.0 for item in self.decades)

        runtime = row.get("runtime")
        community_avg = row.get("community_avg")
        values.append(float(runtime) / 180.0 if pd.notna(runtime) else 0.0)
        values.append(float(community_avg) / 10.0 if pd.notna(community_avg) else 0.0)
        return values


def fit_encoder_from_dir(enriched_dir: Path = Path("data/enriched")) -> tuple[FeatureEncoder, dict[str, pd.DataFrame]]:
    frames = load_enriched_user_frames(enriched_dir)
    encoder = FeatureEncoder().fit(frames)
    return encoder, frames
