from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

from cinematch.joint import score_columns
from cinematch.models import train_all_models

warnings.filterwarnings("ignore", category=UserWarning, module=r"letterboxdpy.*")

DEFAULT_ENRICHED_DIR = Path("data/enriched")
DEFAULT_MODEL_DIR = Path("models")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
_TITLE_YEAR_RE = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)\s*$")
ENRICHED_COLUMNS = [
    "title",
    "year",
    "rating",
    "genres",
    "director",
    "top_cast",
    "runtime",
    "original_language",
    "community_avg",
    "tmdb_id",
]


def log(message: str) -> None:
    print(message, flush=True)


def _friendly_profile_error(username: str, exc: Exception) -> str:
    text = str(exc)
    if '"code": 404' in text or "404" in text:
        return f"Couldn't find user {username!r} on Letterboxd."
    return f"Could not load profile {username!r}: {text.splitlines()[0]}"


def _yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = input(prompt + suffix).strip().casefold()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _load_user_class():
    repo_root = Path(__file__).resolve().parents[1]
    local_pkg_root = repo_root / "letterboxdpy"
    if local_pkg_root.exists():
        sys.path.insert(0, str(local_pkg_root))
    from letterboxdpy.user import User
    return User


def _tmdb_key() -> str:
    key = "a64487d2fb1794713e9ad41159cca6fd"
    if not key:
        raise RuntimeError("TMDB_API_KEY is required in your environment")
    return key


def _parse_movie_input(raw: str) -> tuple[str, int | None]:
    text = raw.strip()
    match = _TITLE_YEAR_RE.match(text)
    if match:
        return match.group("title").strip(), int(match.group("year"))
    return text, None


def _positive_int(s: str) -> int:
    n = int(s)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return n


def _split_movies(values: list[str]) -> list[str]:
    movies: list[str] = []
    for value in values:
        movies.extend([item.strip() for item in value.split(",") if item.strip()])
    return movies


def _tmdb_request(endpoint: str, params: dict) -> dict:
    full_params = {"api_key": _tmdb_key(), **params}
    resp = requests.get(f"{TMDB_BASE_URL}{endpoint}", params=full_params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _search_tmdb_movies(title: str, year: int | None = None, limit: int = 5) -> list[dict]:
    params: dict = {"query": title}
    if year is not None:
        params["year"] = year
    results = _tmdb_request("/search/movie", params=params).get("results", [])
    return results[:limit]


def _fetch_tmdb_enrichment(title: str, year: int | None = None, top_cast: int = 3) -> dict | None:
    params: dict = {"query": title}
    if year is not None:
        params["year"] = year
    search = _tmdb_request("/search/movie", params=params)
    results = search.get("results", [])
    if not results:
        return None

    movie_id = results[0]["id"]
    details = _tmdb_request(f"/movie/{movie_id}", params={"append_to_response": "credits"})
    genres = [item["name"] for item in details.get("genres", [])]
    directors = [
        item["name"]
        for item in details.get("credits", {}).get("crew", [])
        if item.get("job") == "Director"
    ]
    cast = sorted(details.get("credits", {}).get("cast", []), key=lambda item: item.get("order", 999))
    top_cast_names = [item["name"] for item in cast[:top_cast]]
    release_date = details.get("release_date") or ""
    release_year = int(release_date[:4]) if isinstance(release_date, str) and release_date[:4].isdigit() else year

    return {
        "title": details.get("title") or title,
        "year": release_year,
        "genres": "|".join(genres),
        "director": "|".join(directors),
        "top_cast": "|".join(top_cast_names),
        "runtime": details.get("runtime"),
        "original_language": details.get("original_language"),
        "community_avg": details.get("vote_average"),
        "tmdb_id": details.get("id"),
    }


def _rated_movies_for_user(username: str) -> list[dict]:
    User = _load_user_class()
    user = User(username)
    rated_movies: list[dict] = []

    movies_dict = user.get_films().get("movies", {})
    for movie in movies_dict.values():
        if not isinstance(movie, dict):
            continue
        rating = movie.get("rating")
        if rating is None:
            continue
        rated_movies.append(
            {
                "title": movie.get("name") or movie.get("title") or "<unknown title>",
                "year": movie.get("year"),
                "rating": float(rating),
            }
        )

    if rated_movies:
        return rated_movies

    diary_entries = user.get_diary().get("entries", {})
    for entry in diary_entries.values():
        if not isinstance(entry, dict):
            continue
        rating = (entry.get("actions") or {}).get("rating")
        if rating is None:
            continue
        rated_movies.append(
            {
                "title": entry.get("name") or "<unknown title>",
                "year": entry.get("release"),
                "rating": float(rating),
            }
        )

    return rated_movies


def _profile_summary(username: str) -> dict:
    rated_movies = _rated_movies_for_user(username)
    titles = [movie["title"] for movie in rated_movies[:5]]
    return {"rated": len(rated_movies), "titles": titles}


def ensure_enriched_csv(
    username: str,
    enriched_dir: Path,
    force: bool,
    limit: int,
    top_cast: int,
) -> Path:
    out_path = enriched_dir / f"{username}.csv"
    if out_path.exists() and not force:
        log(f"[1/3] {username}: using existing enriched dataset at {out_path}")
        return out_path

    log(f"[1/3] {username}: scraping Letterboxd rated films...")
    rated_movies = _rated_movies_for_user(username)
    if not rated_movies:
        raise RuntimeError(f"{username}: no rated movies found")

    rows: list[dict] = []
    failures = 0
    for movie in rated_movies[: max(0, limit)]:
        enriched = _fetch_tmdb_enrichment(movie["title"], movie.get("year"), top_cast=top_cast)
        if enriched is None:
            failures += 1
            continue
        rows.append({**enriched, "rating": movie["rating"]})

    if not rows:
        raise RuntimeError(f"{username}: no movies could be enriched from TMDB")

    df = pd.DataFrame(rows, columns=ENRICHED_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    log(
        f"[1/3] {username}: saved {len(df)} enriched rated films "
        f"(skipped {failures}) -> {out_path}"
    )
    return out_path


def _candidate_row(enriched: dict) -> pd.DataFrame:
    row = {**enriched, "rating": None}
    return pd.DataFrame([row], columns=ENRICHED_COLUMNS)


def _release_year(match: dict) -> int | None:
    release_date = match.get("release_date")
    if not release_date:
        return None
    try:
        return int(str(release_date)[:4])
    except ValueError:
        return None


def print_movie_summary(movie: pd.DataFrame, heading: str = "Selected movie") -> None:
    row = movie.iloc[0]
    title = row.get("title", "Unknown")
    year = row.get("year")
    year_text = "unknown year" if pd.isna(year) else str(int(float(year)))
    director = ", ".join([part for part in str(row.get("director", "")).split("|") if part]) or "unknown"
    genres = ", ".join([part for part in str(row.get("genres", "")).split("|") if part]) or "unknown"
    print()
    print(f"{heading}: {title} ({year_text})")
    print(f"  Director: {director}")
    print(f"  Genres: {genres}")


def confirm_candidate_movie(raw_movie: str, top_cast: int, match_limit: int = 5) -> pd.DataFrame | None:
    title, year = _parse_movie_input(raw_movie)
    while True:
        if not title:
            candidate = input("Type a movie title, or s to skip: ").strip()
            if candidate.casefold() in {"s", "skip"}:
                return None
            title, year = _parse_movie_input(candidate)
            continue

        matches = _search_tmdb_movies(title, year, limit=match_limit)
        if not matches:
            print(f"No TMDB matches for {title!r}.")
            candidate = input("Type a corrected title, or s to skip: ").strip()
            if candidate.casefold() in {"s", "skip"}:
                return None
            title, year = _parse_movie_input(candidate)
            continue

        print(f"\nTop TMDB matches for {title!r}:")
        for i, match in enumerate(matches, start=1):
            match_title = match.get("title") or "Untitled"
            match_year = _release_year(match) or "unknown year"
            print(f"  {i}. {match_title} ({match_year})")
        print("  Enter: choose 1")
        print("  r: search again with a corrected title")
        print("  s: skip this movie")

        choice = input(f"Choose 1-{len(matches)}, Enter for 1, r, or s: ").strip().casefold()
        if choice in {"s", "skip"}:
            return None
        if choice in {"r", "retry"}:
            corrected = input("Type corrected title: ").strip()
            title, year = _parse_movie_input(corrected)
            continue
        if not choice:
            idx = 1
        elif choice.isdigit() and 1 <= int(choice) <= len(matches):
            idx = int(choice)
        else:
            corrected = input("Invalid choice. Type corrected title or s to skip: ").strip()
            if corrected.casefold() in {"s", "skip"}:
                return None
            title, year = _parse_movie_input(corrected)
            continue

        match = matches[idx - 1]
        enriched = _fetch_tmdb_enrichment(
            title=match.get("title") or title,
            year=_release_year(match) or year,
            top_cast=top_cast,
        )
        if enriched is None:
            print("TMDB enrichment failed for selection, try again.")
            continue
        movie = _candidate_row(enriched)
        print_movie_summary(movie)
        if _yes_no("Use this movie?", default=True):
            return movie
        corrected = input("Type corrected title, or s to skip: ").strip()
        if corrected.casefold() in {"s", "skip"}:
            return None
        title, year = _parse_movie_input(corrected)


def score_candidate_frame(encoder, models, users: list[str], candidates: pd.DataFrame, lam: float) -> pd.DataFrame:
    X = encoder.transform(candidates)
    out = candidates[["title", "year"]].copy()
    prediction_cols = []
    for user in users:
        col = f"{user}_pred"
        prediction_cols.append(col)
        out[col] = models[user].predict(X)
    means, stds, scores = score_columns(out[prediction_cols].to_numpy(), lam=lam)
    out["mean_prediction"] = means
    out["std_prediction"] = stds
    out["group_score"] = scores
    return out.sort_values("group_score", ascending=False).reset_index(drop=True)


def print_scored_results(ranked: pd.DataFrame, users: list[str], lam: float) -> None:
    if len(ranked) == 1:
        print("\nScore result")
        print("------------")
    else:
        print(f"\nRanking {len(ranked)} candidate movies by predicted group score")
        print("----------------------------------------------------------")
    print(f"Group score = mean predicted rating - {lam:g} * disagreement")
    print()

    headers = ["Rank", "Movie", "Year", "Group", "Mean", "Disagree", *users]
    widths = [4, 42, 6, 7, 6, 9, *([8] * len(users))]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        values = [
            str(rank),
            str(row["title"])[:42],
            "" if pd.isna(row["year"]) else str(int(float(row["year"]))),
            f"{row['group_score']:.3f}",
            f"{row['mean_prediction']:.3f}",
            f"{row['std_prediction']:.3f}",
        ]
        values.extend(f"{row[f'{user}_pred']:.2f}" for user in users)
        print("  ".join(value.ljust(width) for value, width in zip(values, widths)))

    top = ranked.iloc[0]
    print()
    print(
        f"Top pick: {top['title']} scores well because the predicted average is "
        f"{top['mean_prediction']:.2f} with disagreement {top['std_prediction']:.2f}."
    )


def _build_candidate_frame(candidates: list[str], top_cast: int) -> pd.DataFrame:
    rows: list[dict] = []
    failures: list[str] = []
    for raw in candidates:
        title, year = _parse_movie_input(raw)
        enriched = _fetch_tmdb_enrichment(title, year, top_cast=top_cast)
        if enriched is None:
            failures.append(raw)
            continue
        rows.append({**enriched, "rating": None})

    if failures:
        log(f"[warn] TMDB could not resolve: {', '.join(failures)}")
    if not rows:
        raise RuntimeError("No candidate movies could be enriched")
    return pd.DataFrame(rows, columns=ENRICHED_COLUMNS).drop_duplicates(subset=["title", "year"])


def interactive_command(args: argparse.Namespace) -> int:
    print("CineMatch interactive recommender")
    print("--------------------------------")

    users: list[str] = []
    while True:
        if not users:
            prompt = "Enter Letterboxd username #1 (or q to quit): "
        else:
            prompt = f"Enter Letterboxd username #{len(users)+1} (Enter or 'done' to finish, q to quit): "
        raw = input(prompt).strip()
        if raw.casefold() in {"q", "quit"}:
            if users:
                break
            return 0
        if not raw or raw.casefold() in {"done", "d"}:
            if users:
                break
            print("Please enter at least one username.")
            continue
        try:
            summary = _profile_summary(raw)
        except Exception as exc:  # noqa: BLE001
            print(_friendly_profile_error(raw, exc))
            print("Try another username.")
            continue
        print(f"Profile check: {raw} ({summary['rated']} rated movies)")
        if summary["rated"] < 10:
            print(f"  {raw} has only {summary['rated']} rated movies; we need at least 10 to learn taste.")
            print("Try another username.")
            continue
        if summary["titles"]:
            print("  Sample titles:")
            for title in summary["titles"]:
                print(f"   - {title}")
        if _yes_no("Use this profile?", default=True):
            users.append(raw)

    print()
    print("Enter candidate movie titles separated by commas.")
    print("Example: Heat (1995), Dune: Part Two, The Social Network")
    raw_movies = input("Movies: ").strip()
    movie_inputs = _split_movies([raw_movies])
    if not movie_inputs:
        print("No candidate movies provided.")
        return 0

    confirmed: list[pd.DataFrame] = []
    for raw_movie in movie_inputs:
        movie = confirm_candidate_movie(raw_movie, top_cast=args.top_cast)
        if movie is not None:
            confirmed.append(movie)
    if not confirmed:
        print("No movies were confirmed, nothing to score.")
        return 0

    for user in users:
        ensure_enriched_csv(
            username=user,
            enriched_dir=args.enriched_dir,
            force=args.force_enrich,
            limit=args.profile_limit,
            top_cast=args.top_cast,
        )

    log("[2/3] Training per-user regression models...")
    encoder, models = train_all_models(
        enriched_dir=args.enriched_dir,
        model_dir=args.model_dir,
        model_kind=args.model_kind,
    )
    missing_models = [user for user in users if user not in models]
    if missing_models:
        raise RuntimeError(f"Missing trained models for: {missing_models}")

    log("[3/3] Scoring confirmed candidate movies...")
    candidates = pd.concat(confirmed, ignore_index=True).drop_duplicates(subset=["title", "year"])
    ranked = score_candidate_frame(encoder, models, users, candidates, lam=args.lam)
    print_scored_results(ranked, users, args.lam)
    return 0


def mutual_command(args: argparse.Namespace) -> int:
    users = list(dict.fromkeys(args.users))
    if not users:
        raise SystemExit("Provide at least one username with --users")

    for user in users:
        try:
            ensure_enriched_csv(
                username=user,
                enriched_dir=args.enriched_dir,
                force=args.force_enrich,
                limit=args.profile_limit,
                top_cast=args.top_cast,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(_friendly_profile_error(user, exc))

    log("[2/3] Training per-user regression models...")
    encoder, models = train_all_models(
        enriched_dir=args.enriched_dir,
        model_dir=args.model_dir,
        model_kind=args.model_kind,
    )
    missing_models = [user for user in users if user not in models]
    if missing_models:
        raise RuntimeError(f"Missing trained models for: {missing_models}")

    candidates = _split_movies(args.movies)
    if not candidates:
        raise SystemExit("Provide at least one candidate movie in --movies")

    log("[3/3] Enriching and scoring candidate movies...")
    candidates_df = _build_candidate_frame(candidates, top_cast=args.top_cast)
    ranked = score_candidate_frame(encoder, models, users, candidates_df, lam=args.lam)
    print_scored_results(ranked, users, args.lam)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cinematch")
    sub = parser.add_subparsers(dest="cmd")

    mutual = sub.add_parser(
        "mutual",
        help="Group recommendation from candidate movie list (one or more users)",
    )
    mutual.add_argument("--users", nargs="+", required=True, metavar="USERNAME", help="One or more Letterboxd usernames")
    mutual.add_argument(
        "--movies",
        nargs="+",
        required=True,
        help='Candidate movies (comma-separated or repeated). Example: --movies "Heat (1995), Dune"',
    )
    mutual.add_argument("--lambda", dest="lam", type=float, default=1.0)
    mutual.add_argument("--model-kind", choices=["ridge", "lasso", "best"], default="best")
    mutual.add_argument("--force-enrich", action="store_true")
    mutual.add_argument("--profile-limit", type=_positive_int, default=300)
    mutual.add_argument("--top-cast", type=int, default=3)
    mutual.add_argument("--enriched-dir", type=Path, default=DEFAULT_ENRICHED_DIR)
    mutual.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    mutual.set_defaults(func=mutual_command)

    interactive = sub.add_parser("interactive", help="Guided prompt flow with validation/confirmation")
    interactive.add_argument("--lambda", dest="lam", type=float, default=1.0)
    interactive.add_argument("--model-kind", choices=["ridge", "lasso", "best"], default="best")
    interactive.add_argument("--force-enrich", action="store_true")
    interactive.add_argument("--profile-limit", type=_positive_int, default=300)
    interactive.add_argument("--top-cast", type=int, default=3)
    interactive.add_argument("--enriched-dir", type=Path, default=DEFAULT_ENRICHED_DIR)
    interactive.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    interactive.set_defaults(func=interactive_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args:
        raw_args = ["interactive"]
    args = build_parser().parse_args(raw_args)
    try:
        return args.func(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCanceled.")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
