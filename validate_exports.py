#!/usr/bin/env python3
"""Command-line utility for validating Letterboxd export files.

The script checks for:
1. Rating mismatches for the same film across diary, ratings, and reviews exports.
2. Duplicate film entries in the reviews export that use different ratings.
3. Duplicate diary entries on the same date for the same film.
4. Deleted diary/review entries that are missing from their corresponding primary exports.
5. Reports a final success message when all checks pass.
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


def _normalise_rating(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
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


def _coerce_rewatch_flag(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().eq("yes")


def _build_film_key(df: pd.DataFrame) -> pd.Series:
    uri = df.get("Letterboxd URI")
    if uri is None:
        uri = pd.Series([""] * len(df), index=df.index)
    uri = uri.fillna("").astype(str).str.strip()

    name = df.get("Name")
    if name is None:
        name = pd.Series([""] * len(df), index=df.index)
    name = name.fillna("").astype(str).str.strip()

    year = df.get("Year")
    if year is None:
        year = pd.Series([""] * len(df), index=df.index)
    year = year.fillna("").astype(str).str.strip()

    fallback = name + " (" + year + ")"
    return uri.where(uri != "", fallback)


def _build_sort_date(df: pd.DataFrame) -> pd.Series:
    watched = pd.to_datetime(df.get("Watched Date"), errors="coerce")
    logged = pd.to_datetime(df.get("Date"), errors="coerce")
    if watched is None:
        return logged
    if logged is None:
        return watched
    return watched.fillna(logged)


def find_rewatch_diary_anomalies(diary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diary = diary.copy()
    diary["film_key"] = _build_film_key(diary)
    diary["sort_date"] = _build_sort_date(diary)
    diary["rewatch_flag"] = _coerce_rewatch_flag(diary.get("Rewatch", pd.Series(index=diary.index)))

    diary = diary.sort_values(["film_key", "sort_date", "Date"])
    diary["watch_index"] = diary.groupby("film_key", dropna=False).cumcount()

    first_entries = diary[diary["watch_index"] == 0]
    first_rewatch = first_entries[first_entries["rewatch_flag"]]

    later_entries = diary[diary["watch_index"] > 0]
    later_not_rewatch = later_entries[~later_entries["rewatch_flag"]]

    cols = ["Name", "Year", "Letterboxd URI", "Date", "Watched Date", "Rewatch"]
    first_rewatch = first_rewatch[[c for c in cols if c in first_rewatch.columns]].sort_values(["Name", "Year"])
    later_not_rewatch = later_not_rewatch[[c for c in cols if c in later_not_rewatch.columns]].sort_values(
        ["Name", "Year", "Date"]
    )

    return first_rewatch.reset_index(drop=True), later_not_rewatch.reset_index(drop=True)


def find_duplicate_diary_entries_same_date(diary: pd.DataFrame) -> pd.DataFrame:
    diary = diary.copy()
    diary["film_key"] = _build_film_key(diary)
    diary["sort_date"] = _build_sort_date(diary)
    dup_mask = diary.duplicated(subset=["film_key", "sort_date"], keep=False)
    duplicates = diary[dup_mask]
    cols = ["Name", "Year", "Letterboxd URI", "Date", "Watched Date", "Rewatch", "Rating", "sort_date"]
    duplicates = duplicates[[c for c in cols if c in duplicates.columns]].sort_values(["Name", "Year", "sort_date"])
    if "sort_date" in duplicates.columns:
        duplicates = duplicates.drop(columns=["sort_date"])
    return duplicates.reset_index(drop=True)




def find_rating_mismatches_across_sources(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    info_map: dict[str, dict[str, str]] = {}
    rating_sets: dict[str, dict[str, list[float]]] = {}

    for label, df in sources.items():
        df = df.copy()
        df["film_key"] = _build_film_key(df)
        df["rating_norm"] = df["Rating"].map(_normalise_rating)
        info = (
            df[["film_key", "Name", "Year", "Letterboxd URI"]]
            .drop_duplicates(subset=["film_key"])
            .set_index("film_key")
        )
        info_map.update(info.fillna("").astype(str).to_dict(orient="index"))
        rating_sets[label] = (
            df.groupby("film_key", dropna=False)["rating_norm"]
            .apply(lambda s: sorted({v for v in s if v is not None}))
            .to_dict()
        )

    all_keys = set().union(*[set(m.keys()) for m in rating_sets.values()])
    records: list[dict[str, str]] = []
    for key in sorted(all_keys):
        per_source = {}
        union: set[float] = set()
        source_internal_mismatch = False
        for label, mapping in rating_sets.items():
            values = mapping.get(key, [])
            if len(values) > 1:
                source_internal_mismatch = True
            union.update(values)
            per_source[label] = ", ".join(str(v) for v in values) if values else ""

        if len(union) <= 1 and not source_internal_mismatch:
            continue

        info = info_map.get(key, {})
        record = {
            "Name": info.get("Name", ""),
            "Year": info.get("Year", ""),
            "Letterboxd URI": info.get("Letterboxd URI", ""),
        }
        for label in sources:
            record[label] = per_source.get(label, "")
        records.append(record)

    if not records:
        return pd.DataFrame(columns=["Name", "Year", "Letterboxd URI", *sources.keys()])
    return pd.DataFrame(records).sort_values(["Name", "Year"], na_position="last")


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
        validate_columns(diary, {"Name", "Rewatch", "Rating"}, "diary export")
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
    rating_mismatches = find_rating_mismatches_across_sources(
        {
            "ratings.csv": ratings,
            "reviews.csv": reviews,
            "diary.csv": diary if diary is not None else pd.DataFrame(columns=["Name", "Year", "Letterboxd URI", "Rating"]),
        }
    )
    dup_reviews = find_duplicate_reviews(reviews)
    all_good = True

    print(f"== Rating mismatches between {ratings_path} and {reviews_path} ==")
    if mismatches.empty:
        print("✅ No rating mismatches found.")
    else:
        print("❌ Rating mismatches found.")
        print(mismatches.to_string(index=False))
        all_good = False

    print("\n== Rating mismatches across ratings/reviews/diary ==")
    if rating_mismatches.empty:
        print("✅ No cross-export rating mismatches found.")
    else:
        print("❌ Cross-export rating mismatches found.")
        print(rating_mismatches.to_string(index=False))
        all_good = False

    print(f"\n== Duplicate Names in {reviews_path} with differing Ratings ==")
    if dup_reviews.empty:
        print("✅ No Names in reviews have multiple different Ratings.")
    else:
        print("❌ Duplicate Names with differing Ratings found.")
        print(dup_reviews.to_string(index=False))
        all_good = False


    if deleted_reviews is not None:
        missing_deleted_reviews = find_missing_deleted_entries(deleted_reviews, reviews)
        print(f"\n== Deleted reviews entries missing from {reviews_path} ==")
        if missing_deleted_reviews.empty:
            print("✅ All deleted reviews are present in the primary reviews export.")
        else:
            print("❌ Missing deleted reviews entries found.")
            print(missing_deleted_reviews.to_string(index=False))
            all_good = False
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
            print("✅ All deleted diary entries are present in the primary diary export.")
        else:
            print("❌ Missing deleted diary entries found.")
            print(missing_deleted_diary.to_string(index=False))
            all_good = False
    elif require_deleted_diary or deleted_diary_path != Path("deleted/diary.csv"):
        # Provide feedback when an explicit deleted diary path was inferred but absent.
        print(f"\nSkipped deleted diary check; file not found at {deleted_diary_path}.")

    if diary is not None:
        _, later_not_rewatch = find_rewatch_diary_anomalies(diary)
        duplicate_diary_entries = find_duplicate_diary_entries_same_date(diary)
        print(f"\n== Diary entries after the first watch that are not marked as rewatches ({diary_path}) ==")
        if later_not_rewatch.empty:
            print("✅ All subsequent diary entries are marked as rewatches.")
        else:
            print("❌ Subsequent diary entries not marked as rewatches found.")
            print(later_not_rewatch.to_string(index=False))
            all_good = False

        print(f"\n== Duplicate diary entries on the same date for the same film ({diary_path}) ==")
        if duplicate_diary_entries.empty:
            print("✅ No duplicate diary entries on the same date.")
        else:
            print("❌ Duplicate diary entries on the same date found.")
            print(duplicate_diary_entries.to_string(index=False))
            all_good = False

    if all_good:
        print("\n✅🎉 All checks passed. Your exports look clean.")


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
