# Hanoi Food Match - CKI

Streamlit demo cho he thong goi y mon an va quan an cho cap doi tai Ha Noi.

## Chay local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Du lieu

App ho tro hai cach nap du lieu:

1. Dat file Excel trong thu muc `data/`.
2. Upload thu cong tren sidebar khi chay app.

Can 2 file:

- Food master: co cac sheet `foods`, `places`, `place_food_map`.
- Rating file: co sheet `user_ratings` hoac `ratings`.

## Cac nang cap CKI

- Context-aware Filtering.
- Cold-start Onboarding cho nguoi dung moi.
- Distance-aware Re-ranking voi mac dinh `w_distance = 0.10`.
- Model Info co thong tin Learning-to-Rank bang Gradient Boosting.
