import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

class Reranker:
    def __init__(self):
        self.model = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)

    def prepare_features(self, candidates: list[str], user_id: str, item_features_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame({'item_id': candidates})
        df = df.merge(item_features_df, on='item_id', how='left').fillna(0)
        return df

    def train(self, train_df: pd.DataFrame, target_col: str):
        X = train_df.drop(columns=['item_id', 'user_id', target_col])
        y = train_df[target_col]
        self.model.fit(X, y)

    def rank(
        self,
        candidates: list[str],
        user_id: str,
        item_features_df: pd.DataFrame,
        top_k: int = 5,
        gru_scores: dict[str, float] | None = None,
    ) -> list[str]:
        if not candidates:
            return []

        features_df = self.prepare_features(candidates, user_id, item_features_df)
        X = features_df.drop(columns=['item_id'])
        lgbm_scores = self.model.predict_proba(X)[:, 1]

        if gru_scores is not None:
            # GRU задаёт персонализированный сигнал (вероятность кластера),
            # LightGBM — качество айтема (популярность, рейтинг).
            # Итоговый скор = произведение двух сигналов.
            g = np.array([gru_scores.get(item, 1e-9) for item in features_df['item_id']])
            features_df['score'] = lgbm_scores * g
        else:
            features_df['score'] = lgbm_scores

        top_items = (
            features_df.sort_values(by='score', ascending=False)
            .head(top_k)['item_id']
            .tolist()
        )
        return top_items
