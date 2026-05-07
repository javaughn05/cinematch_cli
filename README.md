# CineMatch 

CineMatch is a CLI tool that:

1. Reads two users' public Letterboxd ratings
2. Enriches rated films with TMDB metadata
3. Trains per-user regression models
4. Scores candidate movies for both users
5. Ranks candidates by a mutual recommendation score

Mutual score is:

`mean predicted rating - lambda * disagreement`

where disagreement is the standard deviation across user predictions.

## Features

- Interactive guided mode with:
  - profile validation/confirmation
  - movie title disambiguation via TMDB search
  - confirmation prompts before scoring
- Batch mode for one-line command usage
- Supports Ridge, Lasso, or automatic best model selection

## Project Layout

- `cinematch/` - core CLI, feature encoding, modeling, scoring
- `letterboxdpy/` - local Letterboxd scraping library used by CLI
- `data/enriched/` - generated enriched user datasets (gitignored)
- `models/` - trained model artifacts (gitignored)

## Requirements

- Python 3.10+ recommended
- Network access to:
  - `letterboxd.com`
  - `api.themoviedb.org`

## Setup

Clone and enter the project:

```bash
git clone https://github.com/javaughn05/cinematch_cli
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn requests joblib
python -m pip install -r letterboxdpy/requirements.txt
```

## TMDB API Key

The current code reads the TMDB key in `cinematch/cli.py` inside `_tmdb_key()`.

Right now it is hardcoded there. For your own deployment, replace it with your key or switch `_tmdb_key()` to read from an environment variable like `TMDB_API_KEY`.

## Usage

### 1) Interactive mode (recommended)

Runs guided prompts for profile and movie validation:

```bash
python -m cinematch
```


Optional flags:

- `--model-kind ridge|lasso|best` (default: `best`)
- `--lambda <float>` (default: `1.0`)
- `--profile-limit <int>` (default: `300`)
- `--top-cast <int>` (default: `3`)

### 2) Batch mode

Provide users and candidates in one command:

```bash
python -m cinematch mutual \
  --users user_one user_two \
  --movies "Heat (1995), Dune: Part Two, The Social Network" \
  --model-kind best \
  --lambda 1.0
```

You can also pass repeated movie arguments:

```bash
python -m cinematch mutual \
  --users user_one user_two \
  --movies "Heat (1995)" "Dune: Part Two" "The Social Network"
```

## Output

The CLI prints ranked results including:

- per-user predicted rating
- mean predicted rating
- disagreement (std dev)
- final group score

Top-ranked movie is your best mutual recommendation.

## Troubleshooting

- **"No rated movies found"**
  - Confirm the profile is public and has ratings visible.
- **TMDB movie not found**
  - Include year in title, e.g. `The Thing (1982)`.
- **Model quality seems weak**
  - Use profiles with more rated movies.
  - Keep `--model-kind best`.
- **Import errors**
  - Ensure virtual environment is active and dependencies are installed.

## Development Notes

- Generated data and model artifacts are ignored by git (`data/enriched/`, `models/`).
- If you want reproducible training snapshots, keep local backups of those folders.
