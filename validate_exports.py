#!/usr/bin/env python3
"""Command-line utility for validating Letterboxd export files.

The script checks for:
1. Rating mismatches for the same film across ratings and reviews exports.
2. Duplicate film entries in the reviews export that use different ratings.
3. Deleted diary/review entries that are missing from their corresponding primary exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{label} not found at {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - informative crash for malformed CSV
        raise SystemExit(f"Failed to read {label} at {path}: {exc}") from exc


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{label} is missing required columns: {sorted(missing)}")


def find_rating_mismatches(reviews: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        reviews[["Name", "Rating"]],
        ratings[["Name", "Rating"]],
        on="Name",
        how="inner",
        suffixes=("_review", "_rating"),
    )
    return merged[merged["Rating_review"] != merged["Rating_rating"]].sort_values("Name")


def find_duplicate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    dup_names = reviews.groupby("Name")["Rating"].nunique()
    dup_names = dup_names[dup_names > 1].index
    return reviews[reviews["Name"].isin(dup_names)][["Name", "Rating"]].sort_values("Name")


def find_missing_deleted_entries(deleted: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    missing_mask = ~deleted["Name"].isin(primary["Name"])
    return (
        deleted.loc[missing_mask, ["Name"]]
        .drop_duplicates()
        .sort_values("Name")
        .reset_index(drop=True)
    )


def run_checks(
    reviews_path: Path,
    ratings_path: Path,
    diary_path: Path,
    deleted_reviews_path: Path,
    deleted_diary_path: Path,
    *,
    require_diary: bool,
    require_deleted_reviews: bool,
    require_deleted_diary: bool,
) -> None:
    required_rating_cols = {"Name", "Rating"}

    reviews = load_csv(reviews_path, "reviews export")
    validate_columns(reviews, required_rating_cols, "reviews export")

    ratings = load_csv(ratings_path, "ratings export")
    validate_columns(ratings, required_rating_cols, "ratings export")

    diary: Optional[pd.DataFrame] = None
    if diary_path.exists():
        diary = load_csv(diary_path, "diary export")
        validate_columns(diary, {"Name"}, "diary export")
    elif require_diary:
        raise SystemExit(f"diary export not found at {diary_path}")

    deleted_reviews: Optional[pd.DataFrame] = None
    if deleted_reviews_path.exists():
        deleted_reviews = load_csv(deleted_reviews_path, "deleted reviews export")
        validate_columns(deleted_reviews, {"Name"}, "deleted reviews export")
    elif require_deleted_reviews:
        raise SystemExit(f"deleted reviews export not found at {deleted_reviews_path}")

    deleted_diary: Optional[pd.DataFrame] = None
    if deleted_diary_path.exists():
        deleted_diary = load_csv(deleted_diary_path, "deleted diary export")
        validate_columns(deleted_diary, {"Name"}, "deleted diary export")
    elif require_deleted_diary:
        raise SystemExit(f"deleted diary export not found at {deleted_diary_path}")

    mismatches = find_rating_mismatches(reviews, ratings)
    dup_reviews = find_duplicate_reviews(reviews)

    print(f"== Rating mismatches between {ratings_path} and {reviews_path} ==")
    if mismatches.empty:
        print("No rating mismatches found.")
    else:
        print(mismatches.to_string(index=False))

    print(f"\n== Duplicate Names in {reviews_path} with differing Ratings ==")
    if dup_reviews.empty:
        print("No Names in reviews have multiple different Ratings.")
    else:
        print(dup_reviews.to_string(index=False))

    if deleted_reviews is not None:
        missing_deleted_reviews = find_missing_deleted_entries(deleted_reviews, reviews)
        print(f"\n== Deleted reviews entries missing from {reviews_path} ==")
        if missing_deleted_reviews.empty:
            print("All deleted reviews are present in the primary reviews export.")
        else:
            print(missing_deleted_reviews.to_string(index=False))
    elif deleted_reviews_path != Path("deleted/reviews.csv"):
        print(f"\nSkipped deleted reviews check; file not found at {deleted_reviews_path}.")

    if deleted_diary is not None:
        if diary is None:
            raise SystemExit(
                "Cannot validate deleted diary entries because the primary diary export could not be loaded."
            )
        missing_deleted_diary = find_missing_deleted_entries(deleted_diary, diary)
        print(f"\n== Deleted diary entries missing from {diary_path} ==")
        if missing_deleted_diary.empty:
            print("All deleted diary entries are present in the primary diary export.")
        else:
            print(missing_deleted_diary.to_string(index=False))
    elif require_deleted_diary or deleted_diary_path != Path("deleted/diary.csv"):
        # Provide feedback when an explicit deleted diary path was inferred but absent.
        print(f"\nSkipped deleted diary check; file not found at {deleted_diary_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Letterboxd exports for rating consistency and deleted entry integrity."
    )
    parser.add_argument(
        "--reviews",
        default=None,
        help="Path to the primary reviews export CSV (default: search in letterboxd-*-utc/ then current dir)",
    )
    parser.add_argument(
        "--ratings",
        default=None,
        help="Path to the ratings export CSV (default: search in letterboxd-*-utc/ then current dir)",
    )
    parser.add_argument(
        "--diary",
        default=None,
        help="Path to the primary diary export CSV (default: search in letterboxd-*-utc/ then current dir)",
    )
    parser.add_argument(
        "--deleted-reviews",
        default=None,
        help=(
            "Path to the deleted reviews export CSV to verify against the primary reviews export "
            "(default: search in letterboxd-*-utc/ then current dir)"
        ),
    )
    parser.add_argument(
        "--deleted-diary",
        default=None,
        help=(
            "Path to the deleted diary export CSV to verify against the primary diary export "
            "(default: search in letterboxd-*-utc/ then current dir)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def find_in_letterboxd_dirs(name: str) -> Optional[Path]:
        roots = list(Path.cwd().glob("letterboxd-*-utc/"))
        candidates: list[Path] = []
        for root in roots:
            candidate = root / name
            if candidate.exists():
                candidates.append(candidate)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
        return candidates[0]

    def resolve_path(cli_value: Optional[str], default_name: str) -> tuple[Path, bool]:
        if cli_value:
            return Path(cli_value), True
        found = find_in_letterboxd_dirs(default_name)
        if found:
            return found, False
        return Path(default_name), False

    reviews_path, _ = resolve_path(args.reviews, "reviews.csv")
    ratings_path, _ = resolve_path(args.ratings, "ratings.csv")
    diary_path, diary_explicit = resolve_path(args.diary, "diary.csv")
    deleted_reviews_path, deleted_reviews_explicit = resolve_path(args.deleted_reviews, "deleted/reviews.csv")
    deleted_diary_path, deleted_diary_explicit = resolve_path(args.deleted_diary, "deleted/diary.csv")

    run_checks(
        reviews_path,
        ratings_path,
        diary_path,
        deleted_reviews_path,
        deleted_diary_path,
        require_diary=diary_explicit,
        require_deleted_reviews=deleted_reviews_explicit,
        require_deleted_diary=deleted_diary_explicit,
    )


if __name__ == "__main__":
    main()
