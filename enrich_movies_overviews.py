"""Add checked English TMDB overviews to the MovieLens 1M movie table.

Inputs are the 2017 "The Movies Dataset" files ``links.csv`` and
``movies_metadata.csv``. The base catalogue comes directly from MovieLens 1M movies.dat.
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd
from langdetect import DetectorFactory, detect_langs


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external_data" / "the_movies_dataset"
DetectorFactory.seed = 0


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r", (the|a|an)$", lambda m: m.group(1) + " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def detect_overview_language(value: str) -> tuple[str, float]:
    # This 23-character description is unambiguously English, but the
    # statistical detector classifies it as Norwegian because it is so short.
    if value == "Film from Stephen Kijak":
        return "en_manual", 1.0
    try:
        best = detect_langs(value)[0]
        return best.lang, float(best.prob)
    except Exception:
        return "unknown", 0.0


def main() -> None:
    movies = pd.read_csv(
        ROOT / "ml-1m" / "movies.dat", sep="::", engine="python",
        header=None, names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)$")[0].astype(int)
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
    links = pd.read_csv(
        SOURCE / "links.csv",
        dtype={"movieId": "Int64", "imdbId": "string", "tmdbId": "Int64"},
    )
    meta = pd.read_csv(
        SOURCE / "movies_metadata.csv",
        low_memory=False,
        dtype="string",
        usecols=["id", "imdb_id", "title", "release_date", "overview"],
    )
    meta["tmdb_id_num"] = pd.to_numeric(meta["id"], errors="coerce").astype("Int64")
    meta = meta[meta["tmdb_id_num"].notna()].copy()
    # Repeated TMDB IDs in this snapshot have identical metadata.
    meta = meta.drop_duplicates("tmdb_id_num", keep="first")
    meta["imdb_norm"] = (
        meta["imdb_id"].str.extract(r"tt(\d+)", expand=False).str.lstrip("0")
    )
    links["imdb_norm"] = links["imdbId"].str.lstrip("0")

    joined = movies.merge(
        links[["movieId", "imdbId", "imdb_norm", "tmdbId"]],
        left_on="movie_id", right_on="movieId", how="left", validate="one_to_one",
    ).merge(
        meta[["tmdb_id_num", "imdb_norm", "title", "release_date", "overview", "imdb_id"]],
        left_on="tmdbId", right_on="tmdb_id_num", how="left",
        validate="many_to_one", suffixes=("_ml", "_tmdb"),
    )
    assert len(joined) == len(movies) == movies["movie_id"].nunique()

    imdb_agrees = (
        joined["imdb_norm_ml"].notna()
        & joined["imdb_norm_ml"].eq(joined["imdb_norm_tmdb"])
    ).fillna(False)
    joined["overview_match_method"] = pd.NA
    joined.loc[imdb_agrees, "overview_match_method"] = "movieId+tmdbId+imdbId"
    joined["selected_tmdb_id"] = joined["tmdbId"].where(imdb_agrees)
    joined["selected_title"] = joined["title_tmdb"].where(imdb_agrees)
    joined["selected_date"] = joined["release_date"].where(imdb_agrees)
    joined["selected_overview"] = joined["overview"].where(imdb_agrees)

    # Fallback is deliberately narrow: only records with no IMDb ID in links,
    # and one unique TMDB record with exactly the same title and year.
    meta["title_key"] = meta["title"].map(normalize_title)
    meta["year"] = pd.to_numeric(meta["release_date"].str[:4], errors="coerce").astype("Int64")
    unique_title_year = meta.drop_duplicates(["title_key", "year"], keep=False)
    fallback = joined.loc[
        ~imdb_agrees & joined["imdb_norm_ml"].isna(),
        ["movie_id", "title_clean", "release_year"],
    ].copy()
    fallback["title_key"] = fallback["title_clean"].map(normalize_title)
    fallback = fallback.merge(
        unique_title_year[["title_key", "year", "tmdb_id_num", "title", "release_date", "overview"]],
        left_on=["title_key", "release_year"],
        right_on=["title_key", "year"],
        how="left", validate="many_to_one",
    ).set_index("movie_id")
    matched_fallback = fallback[fallback["tmdb_id_num"].notna()]
    for movie_id, row in matched_fallback.iterrows():
        idx = joined.index[joined["movie_id"] == movie_id][0]
        joined.at[idx, "selected_tmdb_id"] = row["tmdb_id_num"]
        joined.at[idx, "selected_title"] = row["title"]
        joined.at[idx, "selected_date"] = row["release_date"]
        joined.at[idx, "selected_overview"] = row["overview"]
        joined.at[idx, "overview_match_method"] = "unique_exact_title_year"

    joined["overview_status"] = "no_verified_match"
    conflict = ~imdb_agrees & joined["tmdbId"].notna() & joined["imdb_norm_ml"].notna()
    joined.loc[conflict, "overview_status"] = "imdb_id_conflict"
    has_match = joined["selected_tmdb_id"].notna()
    joined.loc[has_match, "overview_status"] = "no_overview"
    joined["overview_en"] = pd.NA
    joined["overview_detected_language"] = pd.NA
    joined["overview_language_probability"] = pd.NA

    for n, (idx, row) in enumerate(joined[has_match].iterrows(), start=1):
        value = row["selected_overview"]
        if pd.notna(value) and str(value).strip():
            value = str(value).strip()
            lang, probability = detect_overview_language(value)
            joined.at[idx, "overview_detected_language"] = lang
            joined.at[idx, "overview_language_probability"] = round(probability, 4)
            if lang in {"en", "en_manual"} and probability >= 0.8:
                joined.at[idx, "overview_en"] = value
                joined.at[idx, "overview_status"] = "ok"
            else:
                joined.at[idx, "overview_status"] = "non_english_or_uncertain"
        if n % 500 == 0:
            print(f"Проверено описаний: {n}/{int(has_match.sum())}", flush=True)

    joined["overview_source_url"] = joined["selected_tmdb_id"].map(
        lambda x: f"https://www.themoviedb.org/movie/{int(x)}" if pd.notna(x) else pd.NA
    )
    extra_columns = [
        "selected_tmdb_id", "selected_title", "selected_date", "overview_en",
        "overview_status", "overview_match_method", "overview_detected_language",
        "overview_language_probability", "overview_source_url",
    ]
    original_columns = ["title_ml" if c == "title" else c for c in movies.columns]
    output = joined[original_columns + extra_columns].rename(columns={
        "title_ml": "title",
        "selected_tmdb_id": "tmdb_id",
        "selected_title": "tmdb_title",
        "selected_date": "tmdb_release_date",
    })
    issues = joined[joined["overview_status"] != "ok"][
        ["movie_id", "title_ml", "release_year", "overview_status", "imdbId",
         "tmdbId", "title_tmdb", "release_date", "imdb_id", "selected_tmdb_id",
         "overview_detected_language", "overview_source_url"]
    ].rename(columns={
        "title_ml": "movielens_title",
        "imdbId": "linked_imdb_id",
        "tmdbId": "linked_tmdb_id",
        "title_tmdb": "linked_tmdb_title",
        "release_date": "linked_tmdb_release_date",
        "imdb_id": "linked_tmdb_imdb_id",
        "selected_tmdb_id": "verified_tmdb_id",
    })
    output_path = ROOT / "movies_with_overviews.csv"
    issues_path = ROOT / "movies_overview_issues.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
    counts = output["overview_status"].value_counts().to_dict()
    print(f"Сохранено {len(output)} фильмов: {output_path}")
    print(f"Требуют внимания: {len(issues)} фильмов: {issues_path}")
    print(f"Статусы: {counts}")


if __name__ == "__main__":
    main()
