# Letterboxd Analysis Tools

Utilities for inspecting your Letterboxd export locally. The scripts stay lightweight, rely on familiar Python tooling, and avoid uploading your personal data.

## Features

- Validate rating consistency across diary, ratings, and reviews exports.
- Detect duplicate review ratings for the same film.
- Detect duplicate diary entries on the same date for the same film.
- Verify deleted diary/review entries still exist in the primary exports.
- Check that all diary entries after the first watch are marked as rewatches.
- Print a final success line when all checks pass.
- Enrich ratings with public averages and generate comparison plots.

## Getting Started

1. **Python**: Install Python 3.11 or newer.

2. **Environment**: Create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Dependencies**: Install the required Python packages.

   ```bash
   pip install -r requirements.txt
   ```

   If you use uv:

   ```bash
   uv pip install -r requirements.txt
   ```

## Prepare Your Data

1. Export your diary from [Letterboxd settings](https://letterboxd.com/settings/data/).
2. Extract the archive. You can either:

   - place the extracted archive directory (it will be named like `letterboxd-<username>-<date>-<time>-utc`) in the repository root — the scripts will automatically discover CSVs inside the most recent `letterboxd-*-utc/` directory; or
   - point the scripts directly at any CSV using the `--input`, `--reviews` or `--ratings` flags.

3. Do not commit the extracted CSVs into the repository — they contain your personal data. The `.gitignore` is configured to ignore `letterboxd-*-utc/` archive directories by default.

## Usage

### Compare personal ratings to the public average

```bash
python compare_letterboxd.py \
   --output artifacts/ratings_enriched.csv
```

By default the script will search for `ratings.csv` inside the most recent `letterboxd-*-utc/` archive in the repo root and write `artifacts/ratings_enriched.csv`. You can override the input using `--input <path>`.

The script generates:

- `artifacts/scatter.png` – your rating vs the public weighted average
- `artifacts/residuals_top.png` – largest absolute residuals

Key options:

Run `python compare_letterboxd.py --help` to see the full CLI.

### Check for rating discrepancies between exports

```bash
python validate_exports.py
```

By default this will search for `reviews.csv`, `ratings.csv`, `diary.csv`, and the `deleted/*.csv` exports inside the most recent `letterboxd-*-utc/` archive, falling back to files in the current directory if none are found. Use `--reviews`, `--ratings`, `--diary`, `--deleted-reviews`, or `--deleted-diary` to point at specific paths.

This reports rating mismatches across diary/ratings/reviews, duplicate review ratings, duplicate diary entries on the same date, deleted diary/review entries missing from primary exports, diary rewatch consistency, and a final success line when all checks pass.

## Repository Layout

- `artifacts/scatter.png` – scatter plot output from the comparison script.
- `artifacts/residuals_top.png` – residual plot output from the comparison script.

## Friendly Scraping Notice

Letterboxd does not provide a public API for these statistics. Keep request rates gentle (the default delay is conservative), and cache the enriched CSV locally so you avoid unnecessary repeat scraping.

## Next Steps

Ideas for future additions include computing taste drift over time, building a rewatch radar, and surfacing underrated performers based on your personal averages. Contributions are welcome once the repository goes public.
