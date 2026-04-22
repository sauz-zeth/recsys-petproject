import os
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from quantizer import ItemQuantizer
from model import SemanticSeqDataset, GenerativeRecommender, train_model
from reranker import Reranker

def get_metrics(actual, predicted, k=10):
    if not actual or not predicted:
        return 0, 0, 0

    actual_set = set(actual)
    top_k = predicted[:k]

    recall = 1.0 if len(actual_set & set(top_k)) > 0 else 0.0

    mrr = 0.0
    ndcg = 0.0
    for i, p in enumerate(top_k):
        if p in actual_set:
            mrr = 1.0 / (i + 1)
            ndcg = 1.0 / np.log2(i + 2)
            break

    return recall, mrr, ndcg

def evaluate_popularity_baseline(test_data, ratings_df, k=10):
    pop_ranking = (
        ratings_df.groupby('item_id').size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    results = {"recall": [], "mrr": [], "ndcg": []}
    for uid, history, target in test_data:
        seen = set(history)
        recs = [it for it in pop_ranking if it not in seen][:k]
        rec, mrr, ndcg = get_metrics([target], recs, k=k)
        results["recall"].append(rec)
        results["mrr"].append(mrr)
        results["ndcg"].append(ndcg)
    return {m: float(np.mean(v)) for m, v in results.items()}

def download_ml100k():
    if not os.path.exists('ml-100k'):
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        with urllib.request.urlopen(url) as response, zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall()

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=names)
    ratings[['user_id', 'item_id']] = ratings[['user_id', 'item_id']].astype(str)

    item_features = ratings.groupby('item_id').agg(
        popularity=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()

    return ratings, item_features

def main():
    ratings, item_features = download_ml100k()

    print("Подготовка сессий...")
    ratings = ratings.sort_values(['user_id', 'timestamp'])
    user_sessions = ratings.groupby('user_id')['item_id'].apply(list).to_dict()

    train_sessions = []
    test_data = []

    for uid, items in user_sessions.items():
        if len(items) < 5: continue
        train_sessions.append(items[:-1])
        test_data.append((uid, items[:-1], items[-1]))

    print("\n--- ЭТАП 1: Retrieval (Нейросеть) ---")
    quantizer = ItemQuantizer(n_clusters=128)
    quantizer.train_embeddings(train_sessions)
    quantizer.fit_kmeans()

    encoded_train = [quantizer.encode_session(s) for s in train_sessions]
    dataset = SemanticSeqDataset(encoded_train, seq_len=5)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = GenerativeRecommender(vocab_size=128, embed_dim=64, hidden_dim=128)
    train_model(model, loader, epochs=7)
    model.eval()

    print("\n--- ЭТАП 2: Ranking (LightGBM) ---")
    reranker = Reranker()

    pos_samples = ratings[['user_id', 'item_id']].sample(2000, random_state=42).copy()
    pos_samples['target'] = 1
    all_items = item_features['item_id'].unique()
    rng = np.random.default_rng(42)
    neg_samples = pd.DataFrame({
        'user_id': rng.choice(pos_samples['user_id'].values, 2000),
        'item_id': rng.choice(all_items, 2000),
        'target': 0
    })

    train_df = pd.concat([pos_samples, neg_samples]).merge(item_features, on='item_id', how='left').fillna(0)
    reranker.train(train_df, 'target')

    print("\n--- ЭТАП 3: Валидация всей системы ---")
    results = {"recall": [], "mrr": [], "ndcg": []}

    for uid, history, target in test_data:
        input_seq = quantizer.encode_session(history[-4:])
        if len(input_seq) < 4:
            input_seq = [0] * (4 - len(input_seq)) + input_seq

        with torch.no_grad():
            logits = model(torch.tensor([input_seq]))
            probs = torch.softmax(logits, dim=-1).numpy()[0]
            top_codes = torch.topk(logits, 10).indices.numpy()[0]

        # Кандидаты в порядке убывания вероятности кластера (для дедупликации)
        candidates = []
        gru_scores = {}
        for code in top_codes:
            code_score = float(probs[int(code)])
            for item in quantizer.decode_semantic_id(int(code)):
                if item not in gru_scores:
                    candidates.append(item)
                    gru_scores[item] = code_score
        candidates = candidates[:200]

        # LightGBM оценивает качество айтема; GRU — персонализированную релевантность.
        # rank() перемножает оба сигнала для финального ранжирования.
        final_recs = reranker.rank(candidates, uid, item_features, top_k=10, gru_scores=gru_scores)

        rec, mrr, ndcg = get_metrics([target], final_recs, k=10)
        results["recall"].append(rec)
        results["mrr"].append(mrr)
        results["ndcg"].append(ndcg)

    model_metrics = {m: float(np.mean(v)) for m, v in results.items()}

    print("\nСчитаем Popularity Baseline...")
    baseline = evaluate_popularity_baseline(test_data, ratings)

    n = len(test_data)
    w = 62
    print(f"\n{'='*w}")
    print(f"  ОЦЕНКА ({n} пользователей, temporal split)")
    print(f"{'='*w}")
    print(f"  {'Модель':<28} {'Recall@10':>10} {'MRR':>8} {'NDCG@10':>9}")
    print(f"  {'-'*(w-2)}")
    print(f"  {'Popularity Baseline':<28} {baseline['recall']:>10.4f} {baseline['mrr']:>8.4f} {baseline['ndcg']:>9.4f}")
    print(f"  {'Two-Stage (наша модель)':<28} {model_metrics['recall']:>10.4f} {model_metrics['mrr']:>8.4f} {model_metrics['ndcg']:>9.4f}")
    print(f"  {'-'*(w-2)}")
    for m, label in [('recall', 'Recall@10'), ('mrr', 'MRR'), ('ndcg', 'NDCG@10')]:
        lift = (model_metrics[m] / baseline[m] - 1) * 100 if baseline[m] > 0 else 0.0
        sign = '+' if lift >= 0 else ''
        print(f"  {label} lift: {sign}{lift:.1f}%")
    print(f"{'='*w}")

if __name__ == "__main__":
    main()
