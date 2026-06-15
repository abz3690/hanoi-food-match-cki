from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import (
    CONTEXT_KEYWORDS,
    HanoiFoodRecommender,
    find_data_files,
    load_data,
)


st.set_page_config(
    page_title="Hanoi Food Match - CKI",
    page_icon="🍽️",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1180px; }
    h1, h2, h3 { color: #991b1b; }
    div[data-testid="stMetricValue"] { color: #991b1b; }
    .small-note {
        padding: 0.75rem 0.9rem;
        border-left: 4px solid #b91c1c;
        background: #fff1f2;
        border-radius: 6px;
        color: #374151;
        margin: 0.4rem 0 1rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def find_default_data():
    local = Path(__file__).parent / "data"
    return find_data_files(local)


@st.cache_resource(show_spinner=False)
def build_recommender_from_paths(food_master_path: str, rating_path: str):
    data = load_data(food_master_path, rating_path)
    return HanoiFoodRecommender(data)


@st.cache_resource(show_spinner=False)
def build_recommender_from_uploads(food_master_file, rating_file):
    data = load_data(food_master_file, rating_file)
    return HanoiFoodRecommender(data)


def show_food_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Chưa tạo được danh sách món phù hợp. Cậu thử nới điều kiện lọc hoặc chọn người dùng khác nhé.")
        return

    cols = [
        c for c in [
            "dish_name", "cuisine_group", "dish_type", "spicy_level",
            "hybrid_score", "context_score", "context_aware_score",
            "cold_start_score", "final_food_score",
        ]
        if c in df.columns
    ]
    show = df[cols].copy()
    rename = {
        "dish_name": "Món ăn",
        "cuisine_group": "Nhóm ẩm thực",
        "dish_type": "Loại món",
        "spicy_level": "Mức cay",
        "hybrid_score": "Điểm Hybrid",
        "context_score": "Điểm ngữ cảnh",
        "context_aware_score": "Điểm sau ngữ cảnh",
        "cold_start_score": "Điểm cold-start",
        "final_food_score": "Điểm chung cặp đôi",
    }
    st.dataframe(show.rename(columns=rename), use_container_width=True, hide_index=True)


def show_place_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("Chưa tìm được quán phù hợp với danh sách món.")
        return

    show = df.copy()
    if "distance_km" in show.columns:
        show["distance_km"] = show["distance_km"].round(2)
    for col in ["place_score_gki", "place_score_cki"]:
        if col in show.columns:
            show[col] = show[col].round(4)

    rename = {
        "place_name": "Quán ăn",
        "district": "Khu vực",
        "matched_food_count": "Số món khớp",
        "avg_rating": "Đánh giá",
        "review_count": "Lượt đánh giá",
        "distance_km": "Khoảng cách (km)",
        "place_score_gki": "Điểm GKI",
        "place_score_cki": "Điểm CKI",
    }
    show = show[[c for c in rename if c in show.columns]].rename(columns=rename)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown(
        """
        <div class="small-note">
        Điểm xếp hạng quán được tính từ mức độ phù hợp với món ăn, đánh giá quán,
        số lượt đánh giá và khoảng cách tới khu vực người dùng lựa chọn.
        </div>
        """,
        unsafe_allow_html=True,
    )


def preference_form(prefix: str, rec: HanoiFoodRecommender):
    st.markdown(f"**{prefix}**")
    cuisines = st.multiselect(
        "Nhóm ẩm thực yêu thích",
        rec.cuisines(),
        key=f"{prefix}_cuisine",
    )
    dish_types = st.multiselect(
        "Loại món yêu thích",
        rec.dish_types(),
        key=f"{prefix}_dish",
    )
    max_spicy = st.slider(
        "Độ cay",
        min_value=0,
        max_value=5,
        value=3,
        key=f"{prefix}_spicy",
    )
    food_options = rec.foods[["food_id", "dish_name"]].copy()
    food_options["label"] = food_options["food_id"] + " - " + food_options["dish_name"]
    chosen = st.multiselect(
        "Một vài món đã thích",
        food_options["label"].tolist(),
        key=f"{prefix}_liked",
        max_selections=5,
    )
    liked_ids = [item.split(" - ", 1)[0] for item in chosen]
    return {
        "liked_food_ids": liked_ids,
        "preferred_cuisines": cuisines,
        "preferred_dish_types": dish_types,
        "max_spicy": max_spicy,
    }


st.title("Hanoi Food Match · CKI")

with st.sidebar:
    st.header("Dữ liệu")
    default_food, default_rating = find_default_data()

    if default_food and default_rating:
        st.success("Đã tìm thấy dữ liệu trong thư mục data.")
        use_upload = st.toggle("Upload file khác", value=False)
    else:
        st.info("Chưa có dữ liệu sẵn. Hãy upload 2 file Excel để chạy app.")
        use_upload = True

    food_upload = rating_upload = None
    if use_upload:
        food_upload = st.file_uploader(
            "Food master: foods, places, place_food_map",
            type=["xlsx", "xls"],
        )
        rating_upload = st.file_uploader(
            "Rating file: users, user_ratings/rated_only",
            type=["xlsx", "xls"],
        )

    st.divider()
    st.header("Thiết lập demo")
    top_food_k = st.slider("Số món gợi ý", 1, 10, 5)
    top_place_k = st.slider("Số quán gợi ý", 1, 10, 5)


try:
    if use_upload:
        if food_upload is None or rating_upload is None:
            st.stop()
        recommender = build_recommender_from_uploads(food_upload, rating_upload)
    else:
        recommender = build_recommender_from_paths(str(default_food), str(default_rating))
except Exception as exc:
    st.error(f"Không đọc được dữ liệu: {exc}")
    st.stop()


tab_recommend, tab_model, tab_about = st.tabs([
    "Gợi ý món & quán",
    "Model Info",
    "Hướng dẫn demo",
])

with tab_recommend:
    c1, c2, c3 = st.columns([1.2, 1.2, 1])

    with c1:
        group_mode = st.radio(
            "Chế độ người dùng",
            ["Đã có lịch sử", "Chưa có lịch sử"],
            horizontal=True,
        )
        strategy = st.selectbox(
            "Chiến lược gộp sở thích cặp đôi",
            ["Average", "Least Misery", "Most Pleasure"],
            index=0,
        )

    with c2:
        contexts = list(CONTEXT_KEYWORDS)
        meal_context = st.selectbox("Ngữ cảnh bữa ăn", contexts, index=2)
        occasion = st.selectbox("Tình huống ăn uống", contexts, index=3)
        context_weight = 0.35

    with c3:
        district_options = recommender.districts()
        default_district_index = district_options.index("Hoàn Kiếm") if "Hoàn Kiếm" in district_options else 0
        user_district = st.selectbox("Bạn đang ở khu vực nào?", district_options, index=default_district_index)
        user_location = st.selectbox("Điểm mốc gần bạn", recommender.locations(), index=0)
        w_distance = st.slider(
            "Mức ưu tiên quán gần",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.05,
        )

    st.divider()

    if group_mode == "Đã có lịch sử":
        users = recommender.user_ids()
        if len(users) < 2:
            st.warning("Dữ liệu cần ít nhất 2 người dùng để demo chế độ cặp đôi.")
            st.stop()

        u1, u2 = st.columns(2)
        with u1:
            user_a = st.selectbox("Người thứ nhất", users, index=0)
        with u2:
            user_b = st.selectbox("Người thứ hai", users, index=1)

        food_result = recommender.group_existing_recommend(
            user_a=user_a,
            user_b=user_b,
            strategy=strategy,
            top_k=top_food_k,
            meal_context=meal_context,
            occasion=occasion,
            max_spicy=None,
            context_weight=context_weight,
        )
    else:
        st.markdown("### Onboarding sở thích cho từng người")
        p1, p2 = st.columns(2)
        with p1:
            pref_a = preference_form("Người thứ nhất", recommender)
        with p2:
            pref_b = preference_form("Người thứ hai", recommender)

        no_pref_a = not pref_a["liked_food_ids"] and not pref_a["preferred_cuisines"] and not pref_a["preferred_dish_types"]
        no_pref_b = not pref_b["liked_food_ids"] and not pref_b["preferred_cuisines"] and not pref_b["preferred_dish_types"]
        if no_pref_a or no_pref_b:
            st.info("Bạn chưa chọn sở thích. Hệ thống sẽ ưu tiên các món được 60 người khảo sát đánh giá cao.")

        food_result = recommender.group_cold_start_recommend(
            pref_a=pref_a,
            pref_b=pref_b,
            strategy=strategy,
            top_k=top_food_k,
        )

    st.subheader("Top món ăn phù hợp cho cặp đôi")
    show_food_table(food_result)

    top_food_ids = food_result["food_id"].astype(str).tolist() if food_result is not None and not food_result.empty else []
    place_result = recommender.recommend_places_distance_aware(
        top_food_ids=top_food_ids,
        user_district=user_district,
        user_location_name=user_location,
        w_distance=w_distance,
        top_k=top_place_k,
    )

    st.subheader("Top quán ăn sau Distance-aware Re-ranking")
    show_place_table(place_result)


with tab_model:
    st.subheader("Thông tin mô hình")
    st.markdown(
        """
        App demo sử dụng Hybrid Recommendation làm lõi gợi ý món ăn, sau đó bổ sung
        Context-aware Filtering, Cold-start Onboarding và Distance-aware Re-ranking.
        Learning-to-Rank bằng Gradient Boosting được trình bày như kết quả thực nghiệm
        từ notebook CKI để hỗ trợ phần vấn đáp.
        """
    )

    summary_df, feature_df = recommender.model_info_tables()
    st.markdown("### So sánh các mô hình xếp hạng quán ăn")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("### Đặc trưng đầu vào của LTR Gradient Boosting")
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("w_distance mặc định", "0.10")
    m2.metric("LTR Precision@5", "0.5600")
    m3.metric("LTR AvgDistance@5", "1.5861 km")


with tab_about:
    st.subheader("Cách demo nhanh")
    st.markdown(
        """
        1. Chọn chế độ **Đã có lịch sử** để demo gợi ý cho hai user cũ.
        2. Đổi **Ngữ cảnh bữa ăn** hoặc **Tình huống ăn uống** để minh họa Context-aware Filtering.
        3. Đổi **Điểm mốc gần bạn** và **Mức ưu tiên quán gần** để minh họa Distance-aware Re-ranking.
        4. Chọn **Chưa có lịch sử** để demo Cold-start Onboarding cho cả hai người.
        5. Mở tab **Model Info** khi cần giải thích LTR Gradient Boosting.
        """
    )
