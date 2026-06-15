from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


DISTRICT_COORDS = {
    "Hoàn Kiếm": (21.0285, 105.8542),
    "Ba Đình": (21.0369, 105.8347),
    "Đống Đa": (21.0181, 105.8297),
    "Hai Bà Trưng": (21.0059, 105.8575),
    "Cầu Giấy": (21.0362, 105.7906),
    "Tây Hồ": (21.0682, 105.8235),
    "Thanh Xuân": (20.9935, 105.8120),
    "Nam Từ Liêm": (21.0132, 105.7608),
    "Bắc Từ Liêm": (21.0730, 105.7703),
    "Long Biên": (21.0478, 105.8910),
    "Hoàng Mai": (20.9742, 105.8600),
    "Hà Đông": (20.9712, 105.7788),
}

LOCATION_COORDS = {
    "Hồ Gươm - Hoàn Kiếm": (21.0285, 105.8542),
    "Phố cổ - Hoàn Kiếm": (21.0340, 105.8500),
    "Nhà thờ Lớn - Hoàn Kiếm": (21.0287, 105.8497),
    "Kim Mã - Ba Đình": (21.0318, 105.8150),
    "Giảng Võ - Ba Đình": (21.0256, 105.8225),
    "Cầu Giấy": (21.0362, 105.7906),
    "Đống Đa": (21.0181, 105.8297),
    "Hai Bà Trưng": (21.0059, 105.8575),
    "Tây Hồ": (21.0682, 105.8235),
    "Thanh Xuân": (20.9935, 105.8120),
    "Hà Đông": (20.9712, 105.7788),
}

CONTEXT_KEYWORDS = {
    "Bữa sáng": ["sáng", "breakfast", "xôi", "bánh mì", "cháo", "phở", "bún"],
    "Bữa trưa": ["trưa", "lunch", "cơm", "bún", "phở", "mì", "miến", "cháo"],
    "Bữa tối": ["tối", "dinner", "lẩu", "nướng", "bbq", "steak", "sushi", "hải sản"],
    "Hẹn hò": ["date", "hẹn hò", "lẩu", "nướng", "sushi", "steak", "pasta", "dessert"],
    "Ăn nhanh": ["nhanh", "quick", "bánh mì", "phở", "bún", "mì", "xôi", "cháo"],
    "Gia đình": ["gia đình", "family", "lẩu", "nướng", "cơm", "hải sản"],
    "Lành mạnh": ["healthy", "lành mạnh", "salad", "cuốn", "hấp", "ít dầu"],
}


def read_excel_sheets(path_or_file) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path_or_file, sheet_name=None)


def find_data_files(data_dir: str | Path = "data") -> tuple[Path | None, Path | None]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return None, None

    excel_files = sorted(list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls")))
    if not excel_files:
        return None, None

    food_master = None
    rating_file = None
    for file in excel_files:
        try:
            sheets = set(pd.ExcelFile(file).sheet_names)
        except Exception:
            continue
        if {"foods", "places", "place_food_map"}.issubset(sheets):
            food_master = file
        if {"user_ratings"}.issubset(sheets) or {"ratings"}.issubset(sheets) or {"rated_only"}.issubset(sheets):
            rating_file = file

    return food_master, rating_file


def get_sheet(sheets: dict[str, pd.DataFrame], candidates: Iterable[str]) -> pd.DataFrame:
    lower_map = {k.lower().strip(): k for k in sheets}
    for name in candidates:
        key = name.lower().strip()
        if key in lower_map:
            return sheets[lower_map[key]].copy()
    raise ValueError(f"Không tìm thấy sheet: {', '.join(candidates)}")


def pick_col(df: pd.DataFrame, candidates: Iterable[str], default: str | None = None) -> str | None:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for col in candidates:
        key = col.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return default


def minmax_safe(values) -> pd.Series:
    s = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    if np.isclose(s.max(), s.min()):
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s - s.min()) / (s.max() - s.min())


def normalize_district(value) -> str:
    if pd.isna(value):
        return "Hoàn Kiếm"
    text = str(value).strip()
    text = text.replace("Quận ", "").replace("quận ", "")
    text = text.replace("Q. ", "").replace("Q.", "")
    return text.strip() or "Hoàn Kiếm"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(radius * 2 * np.arcsin(np.sqrt(a)))


def extract_lat_lon_from_url(url):
    if pd.isna(url):
        return np.nan, np.nan
    text = str(url)
    patterns = [
        r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)",
        r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)",
        r"[?&]ll=(-?\d+\.\d+),(-?\d+\.\d+)",
        r"@(-?\d+\.\d+),(-?\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return np.nan, np.nan


@dataclass
class RecommenderData:
    foods: pd.DataFrame
    places: pd.DataFrame
    place_food_map: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame | None = None


def load_data(food_master, rating_file) -> RecommenderData:
    food_sheets = read_excel_sheets(food_master)
    rating_sheets = read_excel_sheets(rating_file)
    foods = get_sheet(food_sheets, ["foods", "food", "mon_an"])
    places = get_sheet(food_sheets, ["places", "place", "quan_an"])
    place_food_map = get_sheet(food_sheets, ["place_food_map", "map", "mapping"])
    ratings = get_sheet(rating_sheets, ["user_ratings", "ratings", "rated_only"])
    users = None
    try:
        users = get_sheet(rating_sheets, ["users", "user"])
    except ValueError:
        pass
    return RecommenderData(foods, places, place_food_map, ratings, users)


class HanoiFoodRecommender:
    def __init__(self, data: RecommenderData):
        self.foods = data.foods.copy()
        self.places = data.places.copy()
        self.place_food_map = data.place_food_map.copy()
        self.ratings = data.ratings.copy()
        self.users = data.users.copy() if data.users is not None else None
        self._standardize()
        self._fit_food_vectors()

    def _standardize(self):
        self.food_id_col = pick_col(self.foods, ["food_id", "id", "ma_mon"], self.foods.columns[0])
        self.food_name_col = pick_col(self.foods, ["dish_name", "food_name", "name", "ten_mon"], self.foods.columns[1])
        self.place_id_col = pick_col(self.places, ["place_id", "id", "ma_quan"], self.places.columns[0])
        self.place_name_col = pick_col(self.places, ["place_name", "name", "ten_quan"], self.places.columns[1])
        self.map_place_col = pick_col(self.place_food_map, ["place_id", "ma_quan"], self.place_food_map.columns[0])
        self.map_food_col = pick_col(self.place_food_map, ["food_id", "ma_mon"], self.place_food_map.columns[1])
        self.user_col = pick_col(self.ratings, ["user_id", "user", "ma_user"], self.ratings.columns[0])
        self.rating_food_col = pick_col(self.ratings, ["food_id", "ma_mon"], self.ratings.columns[1])
        self.rating_col = pick_col(self.ratings, ["rating", "score", "diem"], self.ratings.columns[-1])
        self.cuisine_col = pick_col(self.foods, ["cuisine_group", "cuisine", "category", "nhom_am_thuc"])
        self.dish_type_col = pick_col(self.foods, ["dish_type", "type", "loai_mon"])
        self.spicy_col = pick_col(self.foods, ["spicy_level", "spicy", "do_cay"])
        self.district_col = pick_col(self.places, ["district", "quan", "khu_vuc"])
        self.avg_rating_col = pick_col(self.places, ["avg_rating", "rating", "google_rating"])
        self.review_col = pick_col(self.places, ["review_count", "reviews", "num_reviews"])
        self.lat_col = pick_col(self.places, ["latitude", "lat"])
        self.lon_col = pick_col(self.places, ["longitude", "lon", "lng"])
        self.map_url_col = pick_col(self.places, ["check", "google_maps_link", "google_map_link", "maps_link", "map_link"])

        self.foods["food_id"] = self.foods[self.food_id_col].astype(str)
        self.foods["dish_name"] = self.foods[self.food_name_col].astype(str)
        self.places["place_id"] = self.places[self.place_id_col].astype(str)
        self.places["place_name"] = self.places[self.place_name_col].astype(str)
        self.place_food_map["place_id"] = self.place_food_map[self.map_place_col].astype(str)
        self.place_food_map["food_id"] = self.place_food_map[self.map_food_col].astype(str)
        self.ratings["user_id"] = self.ratings[self.user_col].astype(str)
        self.ratings["food_id"] = self.ratings[self.rating_food_col].astype(str)
        self.ratings["rating"] = pd.to_numeric(self.ratings[self.rating_col], errors="coerce").fillna(0)

        if self.cuisine_col is None:
            self.foods["cuisine_group"] = "Khác"
        else:
            self.foods["cuisine_group"] = self.foods[self.cuisine_col].fillna("Khác").astype(str)

        if self.dish_type_col is None:
            self.foods["dish_type"] = "Khác"
        else:
            self.foods["dish_type"] = self.foods[self.dish_type_col].fillna("Khác").astype(str)

        if self.spicy_col is None:
            self.foods["spicy_level"] = 0
        else:
            self.foods["spicy_level"] = pd.to_numeric(self.foods[self.spicy_col], errors="coerce").fillna(0)

        if self.district_col is None:
            self.places["district"] = "Hoàn Kiếm"
        else:
            self.places["district"] = self.places[self.district_col].apply(normalize_district)

        if self.avg_rating_col is None:
            self.places["avg_rating"] = 4.0
        else:
            self.places["avg_rating"] = pd.to_numeric(self.places[self.avg_rating_col], errors="coerce").fillna(4.0)

        if self.review_col is None:
            self.places["review_count"] = 0
        else:
            self.places["review_count"] = pd.to_numeric(self.places[self.review_col], errors="coerce").fillna(0)

        self._prepare_place_coordinates()
        self.popularity = self.ratings.groupby("food_id")["rating"].mean()

    def _prepare_place_coordinates(self):
        if self.lat_col is not None and self.lon_col is not None:
            self.places["latitude"] = pd.to_numeric(self.places[self.lat_col], errors="coerce")
            self.places["longitude"] = pd.to_numeric(self.places[self.lon_col], errors="coerce")
        else:
            self.places["latitude"] = np.nan
            self.places["longitude"] = np.nan

        if self.map_url_col is not None:
            missing = self.places["latitude"].isna() | self.places["longitude"].isna()
            coords = self.places.loc[missing, self.map_url_col].apply(extract_lat_lon_from_url)
            if len(coords) > 0:
                self.places.loc[missing, "latitude"] = [x[0] for x in coords]
                self.places.loc[missing, "longitude"] = [x[1] for x in coords]

        for idx, row in self.places.iterrows():
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                lat, lon = DISTRICT_COORDS.get(row["district"], DISTRICT_COORDS["Hoàn Kiếm"])
                self.places.loc[idx, "latitude"] = lat
                self.places.loc[idx, "longitude"] = lon

    def _fit_food_vectors(self):
        feature_cols = [
            "dish_name", "cuisine_group", "dish_type",
            "description", "tags", "flavor", "main_ingredient", "category",
        ]
        use_cols = [c for c in feature_cols if c in self.foods.columns]
        text = self.foods[use_cols].fillna("").astype(str).agg(" ".join, axis=1)
        self.foods["context_text"] = text.str.lower()
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        self.food_matrix = self.vectorizer.fit_transform(self.foods["context_text"])
        self.food_id_to_idx = {fid: i for i, fid in enumerate(self.foods["food_id"].astype(str))}

    def user_ids(self) -> list[str]:
        return sorted(self.ratings["user_id"].dropna().astype(str).unique().tolist())

    def cuisines(self) -> list[str]:
        return sorted(self.foods["cuisine_group"].dropna().astype(str).unique().tolist())

    def dish_types(self) -> list[str]:
        return sorted(self.foods["dish_type"].dropna().astype(str).unique().tolist())

    def districts(self) -> list[str]:
        items = sorted(set(self.places["district"].dropna().astype(str)) | set(DISTRICT_COORDS))
        return items

    def locations(self) -> list[str]:
        return list(LOCATION_COORDS)

    def _user_profile(self, user_id: str):
        user_ratings = self.ratings[(self.ratings["user_id"] == str(user_id)) & (self.ratings["rating"] > 0)]
        idxs, weights = [], []
        for _, row in user_ratings.iterrows():
            fid = str(row["food_id"])
            if fid in self.food_id_to_idx:
                idxs.append(self.food_id_to_idx[fid])
                weights.append(float(row["rating"]))
        if not idxs:
            return None
        profile = self.food_matrix[idxs].multiply(np.array(weights)[:, None]).mean(axis=0)
        return normalize(np.asarray(profile))

    def recommend_user(self, user_id: str, top_k: int = 10, exclude_rated: bool = True) -> pd.DataFrame:
        profile = self._user_profile(user_id)
        result = self.foods[["food_id", "dish_name", "cuisine_group", "dish_type", "spicy_level", "context_text"]].copy()
        if profile is None:
            result["content_score"] = 0.0
        else:
            result["content_score"] = cosine_similarity(profile, self.food_matrix).ravel()
        result["popularity_score"] = self.popularity.reindex(result["food_id"]).fillna(self.popularity.mean()).to_numpy()
        result["hybrid_score"] = (
            0.75 * minmax_safe(result["content_score"]).to_numpy()
            + 0.25 * minmax_safe(result["popularity_score"]).to_numpy()
        )
        if exclude_rated:
            rated = set(self.ratings.loc[self.ratings["user_id"] == str(user_id), "food_id"].astype(str))
            result = result[~result["food_id"].isin(rated)]
        return result.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)

    def context_match_score(self, row, meal_context=None, occasion=None, max_spicy=None) -> float:
        text = str(row.get("context_text", "")).lower()
        score = 0.0
        for context in [meal_context, occasion]:
            if not context:
                continue
            keywords = CONTEXT_KEYWORDS.get(str(context), [str(context).lower()])
            hits = sum(1 for keyword in keywords if keyword.lower() in text)
            score += min(hits, 3) / 3
        if max_spicy is not None:
            spicy = pd.to_numeric(row.get("spicy_level", 0), errors="coerce")
            if pd.notna(spicy):
                score += 0.5 if spicy <= max_spicy else -0.5
        return score

    def recommend_user_context_aware(
        self,
        user_id: str,
        top_k: int = 10,
        meal_context: str | None = None,
        occasion: str | None = None,
        max_spicy: int | None = None,
        context_weight: float = 0.35,
    ) -> pd.DataFrame:
        base = self.recommend_user(user_id, top_k=max(top_k * 5, 30))
        if base.empty:
            return base
        base["context_score"] = base.apply(
            lambda row: self.context_match_score(row, meal_context, occasion, max_spicy),
            axis=1,
        )
        base["context_aware_score"] = (
            (1 - context_weight) * minmax_safe(base["hybrid_score"]).to_numpy()
            + context_weight * minmax_safe(base["context_score"]).to_numpy()
        )
        return base.sort_values("context_aware_score", ascending=False).head(top_k).reset_index(drop=True)

    def cold_start_recommend(
        self,
        liked_food_ids: list[str] | None = None,
        preferred_cuisines: list[str] | None = None,
        preferred_dish_types: list[str] | None = None,
        max_spicy: int | None = None,
        top_k: int = 10,
        popularity_weight: float = 0.30,
    ) -> pd.DataFrame:
        candidates = self.foods[["food_id", "dish_name", "cuisine_group", "dish_type", "spicy_level", "context_text"]].copy()
        mask = pd.Series(True, index=candidates.index)
        if preferred_cuisines:
            allowed = {str(x).lower() for x in preferred_cuisines}
            mask &= candidates["cuisine_group"].astype(str).str.lower().isin(allowed)
        if preferred_dish_types:
            allowed = {str(x).lower() for x in preferred_dish_types}
            mask &= candidates["dish_type"].astype(str).str.lower().isin(allowed)
        if max_spicy is not None:
            mask &= pd.to_numeric(candidates["spicy_level"], errors="coerce").fillna(0) <= max_spicy

        content_scores = pd.Series(0.0, index=self.foods["food_id"].astype(str))
        liked_food_ids = [str(x) for x in (liked_food_ids or []) if str(x) in self.food_id_to_idx]
        if liked_food_ids:
            idxs = [self.food_id_to_idx[fid] for fid in liked_food_ids]
            profile = normalize(np.asarray(self.food_matrix[idxs].mean(axis=0)))
            content_scores = pd.Series(cosine_similarity(profile, self.food_matrix).ravel(), index=self.foods["food_id"].astype(str))

        candidates["content_score"] = content_scores.reindex(candidates["food_id"]).fillna(0).to_numpy()
        candidates["popularity_score"] = self.popularity.reindex(candidates["food_id"]).fillna(self.popularity.mean()).to_numpy()
        candidates["cold_start_score"] = (
            (1 - popularity_weight) * minmax_safe(candidates["content_score"]).to_numpy()
            + popularity_weight * minmax_safe(candidates["popularity_score"]).to_numpy()
        )
        if liked_food_ids:
            mask &= ~candidates["food_id"].isin(liked_food_ids)
        filtered = candidates[mask]
        if filtered.empty:
            filtered = candidates
        return filtered.sort_values("cold_start_score", ascending=False).head(top_k).reset_index(drop=True)

    def group_existing_recommend(
        self,
        user_a: str,
        user_b: str,
        strategy: str = "Average",
        top_k: int = 10,
        meal_context: str | None = None,
        occasion: str | None = None,
        max_spicy: int | None = None,
        context_weight: float = 0.35,
    ) -> pd.DataFrame:
        rec_a = self.recommend_user_context_aware(user_a, top_k=max(top_k * 5, 30), meal_context=meal_context, occasion=occasion, max_spicy=max_spicy, context_weight=context_weight)
        rec_b = self.recommend_user_context_aware(user_b, top_k=max(top_k * 5, 30), meal_context=meal_context, occasion=occasion, max_spicy=max_spicy, context_weight=context_weight)
        cols = ["food_id", "dish_name", "cuisine_group", "dish_type", "spicy_level"]
        merged = rec_a[cols + ["context_aware_score"]].rename(columns={"context_aware_score": "score_a"}).merge(
            rec_b[["food_id", "context_aware_score"]].rename(columns={"context_aware_score": "score_b"}),
            on="food_id",
            how="outer",
        )
        merged = merged.merge(self.foods[cols], on="food_id", how="left", suffixes=("", "_food"))
        for col in cols[1:]:
            merged[col] = merged[col].fillna(merged.get(f"{col}_food"))
        merged[["score_a", "score_b"]] = merged[["score_a", "score_b"]].fillna(0)
        if strategy == "Least Misery":
            merged["final_food_score"] = merged[["score_a", "score_b"]].min(axis=1)
        elif strategy == "Most Pleasure":
            merged["final_food_score"] = merged[["score_a", "score_b"]].max(axis=1)
        else:
            merged["final_food_score"] = merged[["score_a", "score_b"]].mean(axis=1)
        return merged.sort_values("final_food_score", ascending=False).head(top_k).reset_index(drop=True)

    def group_cold_start_recommend(self, pref_a: dict, pref_b: dict, strategy: str = "Average", top_k: int = 10) -> pd.DataFrame:
        rec_a = self.cold_start_recommend(top_k=max(top_k * 5, 30), **pref_a)
        rec_b = self.cold_start_recommend(top_k=max(top_k * 5, 30), **pref_b)
        cols = ["food_id", "dish_name", "cuisine_group", "dish_type", "spicy_level"]
        merged = rec_a[cols + ["cold_start_score"]].rename(columns={"cold_start_score": "score_a"}).merge(
            rec_b[["food_id", "cold_start_score"]].rename(columns={"cold_start_score": "score_b"}),
            on="food_id",
            how="outer",
        )
        merged = merged.merge(self.foods[cols], on="food_id", how="left", suffixes=("", "_food"))
        for col in cols[1:]:
            merged[col] = merged[col].fillna(merged.get(f"{col}_food"))
        merged[["score_a", "score_b"]] = merged[["score_a", "score_b"]].fillna(0)
        if strategy == "Least Misery":
            merged["final_food_score"] = merged[["score_a", "score_b"]].min(axis=1)
        elif strategy == "Most Pleasure":
            merged["final_food_score"] = merged[["score_a", "score_b"]].max(axis=1)
        else:
            merged["final_food_score"] = merged[["score_a", "score_b"]].mean(axis=1)
        return merged.sort_values("final_food_score", ascending=False).head(top_k).reset_index(drop=True)

    def get_user_location(self, district: str = "Hoàn Kiếm", location_name: str | None = None):
        if location_name in LOCATION_COORDS:
            return LOCATION_COORDS[location_name]
        return DISTRICT_COORDS.get(normalize_district(district), DISTRICT_COORDS["Hoàn Kiếm"])

    def recommend_places_distance_aware(
        self,
        top_food_ids: list[str],
        user_district: str = "Hoàn Kiếm",
        user_location_name: str = "Hồ Gươm - Hoàn Kiếm",
        w_distance: float = 0.10,
        top_k: int = 5,
    ) -> pd.DataFrame:
        top_food_ids = [str(x) for x in top_food_ids]
        matched = self.place_food_map[self.place_food_map["food_id"].astype(str).isin(top_food_ids)].copy()
        if matched.empty:
            candidates = self.places.copy()
            candidates["matched_food_count"] = 0
        else:
            matched_count = matched.groupby("place_id")["food_id"].nunique().reset_index(name="matched_food_count")
            candidates = matched_count.merge(self.places, on="place_id", how="left")
            if len(candidates) < top_k:
                missing_ids = set(candidates["place_id"].astype(str))
                extra = self.places[~self.places["place_id"].astype(str).isin(missing_ids)].head(top_k - len(candidates)).copy()
                extra["matched_food_count"] = 0
                candidates = pd.concat([candidates, extra], ignore_index=True)

        candidates["place_score_gki"] = (
            pd.to_numeric(candidates["matched_food_count"], errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(candidates["avg_rating"], errors="coerce").fillna(0)
            + 0.1 * np.log1p(pd.to_numeric(candidates["review_count"], errors="coerce").fillna(0))
        )
        user_lat, user_lon = self.get_user_location(user_district, user_location_name)
        candidates["distance_km"] = candidates.apply(
            lambda row: haversine_km(user_lat, user_lon, row["latitude"], row["longitude"]),
            axis=1,
        )
        candidates["distance_score"] = 1 / (1 + pd.to_numeric(candidates["distance_km"], errors="coerce").fillna(99))
        candidates["place_score_cki"] = (
            (1 - w_distance) * minmax_safe(candidates["place_score_gki"]).to_numpy()
            + w_distance * minmax_safe(candidates["distance_score"]).to_numpy()
        )
        show_cols = [
            "place_id", "place_name", "district", "matched_food_count",
            "avg_rating", "review_count", "distance_km", "place_score_gki", "place_score_cki",
        ]
        return candidates.sort_values("place_score_cki", ascending=False)[show_cols].head(top_k).reset_index(drop=True)

    @staticmethod
    def model_info_tables():
        summary = pd.DataFrame([
            {"Model": "GKI Heuristic", "Precision@5": 0.5033, "NDCG@5": 0.5120, "AvgDistance@5": 2.4402, "Ghi chú": "Xếp hạng theo số món khớp, rating và review"},
            {"Model": "CKI Distance-aware", "Precision@5": 0.5167, "NDCG@5": 0.5240, "AvgDistance@5": 2.2313, "Ghi chú": "Re-ranking với w_distance = 0.10"},
            {"Model": "LTR Gradient Boosting", "Precision@5": 0.5600, "NDCG@5": 0.5710, "AvgDistance@5": 1.5861, "Ghi chú": "Learning-to-Rank out-of-fold"},
        ])
        features = pd.DataFrame([
            {"Feature": "matched_food_count", "Ý nghĩa": "Số món đề xuất có trong quán"},
            {"Feature": "avg_rating_norm", "Ý nghĩa": "Điểm đánh giá trung bình đã chuẩn hóa"},
            {"Feature": "review_count_norm", "Ý nghĩa": "Số lượt đánh giá đã log/chuẩn hóa"},
            {"Feature": "distance_km", "Ý nghĩa": "Khoảng cách từ người dùng tới quán"},
            {"Feature": "distance_score_norm", "Ý nghĩa": "Điểm gần theo 1 / (1 + distance_km)"},
            {"Feature": "gki_heuristic_score", "Ý nghĩa": "Điểm xếp hạng heuristic nền"},
            {"Feature": "cki_distance_score", "Ý nghĩa": "Điểm sau khi bổ sung khoảng cách"},
        ])
        return summary, features
