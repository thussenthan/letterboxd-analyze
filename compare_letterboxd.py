#!/usr/bin/env python3
"""Enrich a Letterboxd ratings export with public weighted averages.

Given a CSV containing the columns `Date, Name, Year, Letterboxd URI, Rating`, this
script will scrape the public weighted average for each film, append the results
as new columns, write the enriched CSV, and generate two visualisations:

* `scatter.png` – your rating vs the public average with a y=x reference line
* `residuals_top.png` – films with the largest absolute residuals

The script deduplicates requests, applies polite rate limiting, and caches
successful lookups so subsequent runs are faster.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import unquote

BASE_URL = "https://letterboxd.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LetterboxdCompare/1.2; +https://github.com/thussenthanwalter-angelo)",
    "Accept-Language": "en-US,en;q=0.8",
}
RATING_RE = re.compile(r"Weighted average of\s+([0-9.,]+)", re.IGNORECASE)
AVERAGE_RE = re.compile(r"average rating\s*:?\s*([0-9.,]+)", re.IGNORECASE)

RATING_FROM_URL_RE = re.compile(r"/rated/([^/]+)/")


@dataclass
class FetchContext:
    session: requests.Session
    delay: float
    retries: int
    cache: dict[str, float]
    cache_path: Path | None


def normalise_uri(uri: str) -> str:
    if not isinstance(uri, str):
        return ""
    trimmed = uri.strip()
    if not trimmed:
        return ""
    if trimmed.startswith(("http://", "https://")):
        return trimmed
    if trimmed.startswith("/"):
        return f"{BASE_URL.rstrip('/')}{trimmed}"
    if trimmed.startswith("letterboxd.com"):
        return "https://" + trimmed
    return f"{BASE_URL.rstrip('/')}/{trimmed.lstrip('/')}"


def slug_from_uri(uri: str) -> str | None:
    match = re.search(r"/film/([a-z0-9\-]+)/", uri)
    return match.group(1) if match else None


def parse_user_rating(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        return numeric / 2.0 if numeric > 5 else numeric
    except ValueError:
        pass
    stars = text.count("★")
    half = 0.5 if ("½" in text or ".5" in text) else 0.0
    rating = stars + half
    if rating == 0 and text.count("☆"):
        return 0.0
    return min(max(rating, 0.0), 5.0)


def _coerce_rating(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace("\u00a0", " ")
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"[^0-9.]+", "", cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_from_tooltips(soup: BeautifulSoup) -> float | None:
    for anchor in soup.find_all("a", class_=lambda cls: cls and "display-rating" in cls):
        tooltip = anchor.get("data-original-title") or anchor.get("title")
        rating = _coerce_rating(tooltip)
        if rating is not None:
            return rating
        rating = _coerce_rating(anchor.get("data-average-rating"))
        if rating is not None:
            return rating
        rating = _coerce_rating(anchor.get_text(strip=True))
        if rating is not None:
            return rating
    return None


def _extract_from_meta(soup: BeautifulSoup) -> float | None:
    meta_candidates = [
        ("name", "twitter:data2"),
        ("property", "letterboxd:filmAverageRating"),
        ("name", "letterboxd:filmAverageRating"),
    ]
    for attr, key in meta_candidates:
        node = soup.find("meta", attrs={attr: key})
        if node:
            rating = _coerce_rating(node.get("content"))
            if rating is not None:
                return rating
    return None


def _extract_from_json_ld(soup: BeautifulSoup) -> float | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        stack = [data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                agg = current.get("aggregateRating")
                if isinstance(agg, dict):
                    rating = _coerce_rating(str(agg.get("ratingValue")))
                    if rating is not None:
                        return rating
                stack.extend(current.values())
            elif isinstance(current, list):
                stack.extend(current)
    return None


def extract_public_average(html: str) -> float | None:
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    for extractor in (_extract_from_tooltips, _extract_from_meta, _extract_from_json_ld):
        rating = extractor(soup)  # type: ignore[arg-type]
        if rating is not None:
            return rating

    match = RATING_RE.search(html) or AVERAGE_RE.search(html) or re.search(r'"weightedAverage"\s*:\s*([0-9.,]+)', html)
    if match:
        return _coerce_rating(match.group(1))

    return None


def _rating_from_histogram_token(token: str | None) -> float | None:
    if not token:
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    normalized = (
        cleaned.replace("half-", "0.5")
        .replace("★", "")
        .replace("☆", "")
        .replace("½", ".5")
        .replace(",", ".")
    )
    normalized = normalized.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return _coerce_rating(cleaned)


def extract_average_from_histogram(html: str) -> float | None:
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("div.rating-histogram a.bar")
    if not anchors:
        return None

    total = 0
    weighted = 0.0

    for anchor in anchors:
        href = anchor.get("href") or ""
        token = None
        match = RATING_FROM_URL_RE.search(href)
        if match:
            token = unquote(match.group(1))
        count = None
        title = anchor.get("title") or anchor.get_text(strip=True)
        if title:
            count_match = re.search(r"([0-9][0-9,]*)", title)
            if count_match:
                try:
                    count = int(count_match.group(1).replace(",", ""))
                except ValueError:
                    count = None

        rating = _rating_from_histogram_token(token)
        if rating is None and title:
            tail = title.split(" ratings", 1)[0]
            parts = tail.split()
            if parts:
                rating = _rating_from_histogram_token(parts[-1])

        if rating is None or count is None or count <= 0:
            continue

        weighted += rating * count
        total += count

    if total <= 0:
        return None
    return weighted / total


def request_html(ctx: FetchContext, url: str) -> str | None:
    for attempt in range(1, ctx.retries + 1):
        try:
            response = ctx.session.get(url, timeout=25)
        except requests.RequestException:
            time.sleep(min(3.0, 0.5 * attempt))
            continue
        if response.status_code == 200:
            if ctx.delay:
                time.sleep(ctx.delay)
            return response.text
        if response.status_code in {429, 503, 403}:
            time.sleep(min(6.0, 0.5 * (2 ** attempt)))
            continue
        break
    return None


def fetch_public_average(ctx: FetchContext, uri: str) -> float | None:
    normalised = normalise_uri(uri)
    if not normalised:
        return None

    slug = slug_from_uri(normalised)
    cache_key = slug or normalised.rstrip("/")
    if cache_key in ctx.cache:
        return ctx.cache[cache_key]

    targets: Iterable[str]
    if slug:
        base = f"{BASE_URL.rstrip('/')}/film/{slug}/"
        histogram = f"{BASE_URL.rstrip('/')}/csi/film/{slug}/rating-histogram/"
        targets = (f"{base}json/", histogram, f"{base}ratings/", f"{base}")
    else:
        targets = (normalised,)

    for target in targets:
        html = request_html(ctx, target)
        if target.endswith("json/") and html:
            try:
                payload = json.loads(html)
            except json.JSONDecodeError:
                payload = None
            else:
                rating = _coerce_rating(str(payload.get("averageRating"))) if isinstance(payload, dict) else None
                if rating is None and isinstance(payload, dict):
                    rating = _coerce_rating(str(payload.get("rating")))
                if rating is not None:
                    ctx.cache[cache_key] = rating
                    return rating
        if target.endswith("rating-histogram/"):
            avg = extract_average_from_histogram(html or "")
        else:
            avg = extract_public_average(html or "")
        if avg is not None:
            ctx.cache[cache_key] = avg
            return avg
    return None


def write_cache(ctx: FetchContext) -> None:
    if not ctx.cache_path:
        return
    serialisable = {
        k: v
        for k, v in ctx.cache.items()
        if isinstance(v, (int, float)) and not math.isnan(v)
    }
    ctx.cache_path.write_text(
        json.dumps(serialisable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def ensure_cache(cache_path: Path | None) -> dict[str, float]:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def plot_scatter(df: pd.DataFrame, path: Path) -> None:
    data = df.dropna(subset=["your_rating", "avg_rating_public"])
    if data.empty:
        print("Skipping scatter plot (no comparable data).")
        return
    x = data["avg_rating_public"].to_numpy()
    y = data["your_rating"].to_numpy()
    lo = 0.0
    hi = max(5.0, float(np.nanmax(np.concatenate([x, y]))))

    # Compute metrics
    residuals = y - x
    n = len(x)
    mean_resid = float(np.nanmean(residuals)) if n else float("nan")
    rmse = float(np.sqrt(np.nanmean(residuals ** 2))) if n else float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1]) if n > 1 else float("nan")
    spearman = float(data["your_rating"].corr(data["avg_rating_public"], method="spearman")) if n > 1 else float("nan")

    # R^2 from a linear fit y = m*x + b
    if n > 1:
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        ss_res = np.nansum((y - y_pred) ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        m = b = r_squared = float("nan")

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.6, edgecolors="none")
    # identity reference line
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="#666666")
    # optional regression line (subtle)
    if n > 1 and not math.isnan(m) and not math.isnan(b):
        xs = np.array([lo, hi])
        plt.plot(xs, m * xs + b, linestyle="-", color="#2ca02c", linewidth=1)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Public weighted average (0–5)")
    plt.ylabel("Your rating (0–5)")
    plt.title("Letterboxd ratings: you vs the crowd")

    # Annotate metrics in the plot
    metrics_text = (
        f"N = {n}\n"
        f"R² = {r_squared:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"Mean residual = {mean_resid:.3f}\n"
        f"Pearson r = {pearson:.3f}\n"
        f"Spearman ρ = {spearman:.3f}"
    )
    plt.gca().text(
        0.02,
        0.98,
        metrics_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote scatter plot: {path}")


def plot_residuals(df: pd.DataFrame, path: Path, limit: int) -> None:
    data = df.dropna(subset=["residual", "Name"])
    if data.empty:
        print("Skipping residual plot (no residuals).")
        return
    data = data.assign(abs_residual=data["residual"].abs())
    subset = data.nlargest(limit, "abs_residual").sort_values("abs_residual", ascending=False)
    labels = [f"{name} ({year})" for name, year in zip(subset["Name"], subset["Year"], strict=False)]
    values = subset["residual"].to_numpy()

    plt.figure(figsize=(10, max(4, 0.35 * len(subset))))
    y_pos = np.arange(len(subset))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
    plt.barh(y_pos, values, color=colors)
    plt.axvline(0, linestyle="--", color="#444444")
    plt.yticks(y_pos, labels, fontsize=9)
    plt.gca().invert_yaxis()  # show the largest absolute residual at the top for easier scanning
    plt.xlabel("Residual (your rating − public average)")
    plt.title(f"Top {len(subset)} deviations from the crowd")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Wrote residual plot: {path}")


def compare_letterboxd(
    *,
    input_csv: Path,
    output_csv: Path,
    scatter_path: Path | None,
    residuals_path: Path | None,
    delay: float,
    max_workers: int,
    topn: int,
    cache_path: Path | None,
    raw_output: Path | None,
) -> None:
    df = pd.read_csv(input_csv)
    required = {"Date", "Name", "Year", "Letterboxd URI", "Rating"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Input file {input_csv} is missing required columns: {missing}")

    df["your_rating"] = df["Rating"].map(parse_user_rating)
    targets = df["Letterboxd URI"].map(normalise_uri)

    cache = ensure_cache(cache_path)
    # If we loaded an existing cache file, make that visible to the user
    if cache_path and cache:
        try:
            print(f"Loaded {len(cache)} cached averages from {cache_path}")
        except Exception:
            # Avoid breaking the script if printing fails for any reason
            pass
    ctx = FetchContext(
        session=requests.Session(),
        delay=max(delay, 0.0),
        retries=5,
        cache=cache,
        cache_path=cache_path,
    )
    ctx.session.headers.update(HEADERS)

    unique_targets = sorted({t for t in targets if t})
    print(f"Fetching public averages for {len(unique_targets)} unique films…")

    mem_results: dict[str, float | None] = {}

    def worker(target: str) -> tuple[str, float | None]:
        try:
            return target, fetch_public_average(ctx, target)
        except Exception:
            return target, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(worker, target): target for target in unique_targets}
        for future in tqdm(as_completed(future_map), total=len(future_map), unit="film", desc="Scraping"):
            target, avg = future.result()
            mem_results[target] = avg

    averages = [mem_results.get(t, float('nan')) if t else float('nan') for t in targets]
    df["avg_rating_public"] = pd.to_numeric(averages, errors="coerce")
    df["residual"] = df["your_rating"] - df["avg_rating_public"]

    df_to_write = df.copy()
    if "your_rating" in df_to_write.columns:
        df_to_write = df_to_write.drop(columns=["your_rating"])
    df_to_write["residual"] = df_to_write["residual"].round(4)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_to_write.to_csv(output_csv, index=False)
    if output_csv == input_csv:
        print(f"Updated {output_csv} with public averages.")
    else:
        print(f"Wrote enriched CSV: {output_csv}")

    if raw_output:
        raw_records = (
            pd.DataFrame(
                {
                    "Letterboxd URI": list(mem_results.keys()),
                    "avg_rating_public": [mem_results[k] for k in mem_results],
                }
            )
            .replace({None: float("nan")})
        )
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_records.to_csv(raw_output, index=False)
        print(f"Saved raw scrape results: {raw_output}")

    if scatter_path:
        plot_scatter(df, scatter_path)
    if residuals_path:
        plot_residuals(df, residuals_path, limit=topn)

    matched = df["avg_rating_public"].notna().sum()
    total = len(df)
    print(f"Fetched averages for {matched}/{total} films.")

    missing = df[df["avg_rating_public"].isna()]
    if not missing.empty:
        missing_unique = missing.drop_duplicates(subset=["Letterboxd URI", "Name", "Year"], keep="first")
        print("No public average for:")
        for _, row in missing_unique.iterrows():
            name = str(row.get("Name", "(unknown)")).strip() or "(unknown)"
            year = row.get("Year")
            if pd.notna(year):
                try:
                    year_text = str(int(year))
                except (ValueError, TypeError):
                    year_text = str(year)
                name = f"{name} ({year_text})"
            uri = str(row.get("Letterboxd URI", "")).strip()
            if uri:
                print(f"  - {name} – {uri}")
            else:
                print(f"  - {name}")

    comparable = df.dropna(subset=["your_rating", "avg_rating_public"])
    if not comparable.empty:
        mean_abs = comparable["residual"].abs().mean()
        pearson = comparable[["your_rating", "avg_rating_public"]].corr().iloc[0, 1]
        print(f"Summary statistics | Mean absolute residual: {mean_abs:.3f}\tPearson r: {pearson:.3f}\tN = {len(comparable)}")

        top_residuals = comparable.assign(abs_residual=comparable["residual"].abs()).nlargest(
            50, "abs_residual"
        )
        # Intentionally not writing artifacts/residuals_top.csv per user request.
    else:
        print("No overlapping ratings to summarise.")

    write_cache(ctx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare personal Letterboxd ratings with public averages.")
    parser.add_argument("--input", "-i", default=None, help="Path to the ratings CSV export (default: look in letterboxd-*-utc/ then current dir).")
    parser.add_argument("--output", "-o", default=None, help="Path for the enriched CSV (default: artifacts/<input>_enriched.csv).")
    parser.add_argument("--scatter", default="artifacts/scatter.png", help="Path for the scatter plot, or 'none'.")
    parser.add_argument("--residuals", default="artifacts/residuals_top.png", help="Path for the residual plot, or 'none'.")
    parser.add_argument("--delay", type=float, default=0.10, help="Delay between requests in seconds.")
    parser.add_argument("--max-workers", type=int, default=12, help="Maximum concurrent requests.")
    parser.add_argument("--topn", type=int, default=50, help="Number of films to display in the residual plot.")
    parser.add_argument("--cache", default=".letterboxd_cache.json", help="Path to a JSON cache file (set to 'none' to disable).")
    parser.add_argument("--raw-output", default="none", help="Where to store the unique URI → public rating mapping (use 'none' to skip).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input:
        input_path = Path(args.input)
    else:
    # Prefer CSVs inside letterboxd-*-utc/ archives (most recent), then cwd
        def find_in_letterboxd_dirs(name: str) -> Path | None:
            roots = list(Path.cwd().glob("letterboxd-*-utc/"))
            candidates: list[Path] = []
            for root in roots:
                p = root / name
                if p.exists():
                    candidates.append(p)
            if not candidates:
                return None
            # choose by most recent modification time of the parent folder
            candidates.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
            return candidates[0]

        input_path = find_in_letterboxd_dirs("ratings.csv") or Path("ratings.csv")

    if not input_path.exists():
        raise SystemExit("Input CSV not found. Provide --input or place ratings.csv in data/ or the current directory.")

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("artifacts")
        output_path = output_dir / f"{input_path.stem}_enriched.csv"

    if output_path.resolve() == input_path.resolve():
        raise SystemExit("Refusing to overwrite the input CSV. Choose a different --output path.")

    scatter_path = None if args.scatter.lower() == "none" else Path(args.scatter)
    residuals_path = None if args.residuals.lower() == "none" else Path(args.residuals)
    cache_path = None if args.cache.lower() == "none" else Path(args.cache)
    raw_output = None if args.raw_output.lower() == "none" else Path(args.raw_output)

    compare_letterboxd(
        input_csv=input_path,
        output_csv=output_path,
        scatter_path=scatter_path,
        residuals_path=residuals_path,
        delay=max(args.delay, 0.0),
        max_workers=max(args.max_workers, 1),
        topn=max(args.topn, 1),
        cache_path=cache_path,
        raw_output=raw_output,
    )


if __name__ == "__main__":
    main()
