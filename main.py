import os
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Твои модули
from quantizer import ItemQuantizer
from model import SemanticSeqDataset, GenerativeRecommender, train_model
from reranker import Reranker

def get_metrics(actual, predicted, k=10):
    if not actual or not predicted:
        return 0, 0, 0
    
    actual_set = set(actual)
    top_k = predicted[:k]
    
    # 1. Recall@K
    recall = 1.0 if len(actual_set & set(top_k)) > 0 else 0.0
    
    # 2. MRR и NDCG
    mrr = 0.0
    ndcg = 0.0
    for i, p in enumerate(top_k):
        if p in actual_set:
            mrr = 1.0 / (i + 1)
            ndcg = 1.0 / np.log2(i + 2)
            break # Для одного целевого айтема прерываемся
            
    return recall, mrr, ndcg

def download_ml100k():
    if not os.path.exists('ml-100k'):
        url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        with urllib.request.urlopen(url) as response, zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall()
    
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=names)
    ratings[['user_id', 'item_id']] = ratings[['user_id', 'item_id']].astype(str)
    
    # Фичи товаров для Ранжировщика
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
    test_data = [] # (user_id, history, target_item)

    for uid, items in user_sessions.items():
        if len(items) < 5: continue
        train_sessions.append(items[:-1])
        test_data.append((uid, items[:-1], items[-1]))

    print("\n--- ЭТАП 1: Retrieval (Нейросеть) ---")
    quantizer = ItemQuantizer(n_clusters=64)
    quantizer.train_embeddings(train_sessions)
    quantizer.fit_kmeans()
    
    encoded_train = [quantizer.encode_session(s) for s in train_sessions]
    dataset = SemanticSeqDataset(encoded_train, seq_len=5)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = GenerativeRecommender(vocab_size=64, embed_dim=32, hidden_dim=64)
    train_model(model, loader, epochs=3)

    print("\n--- ЭТАП 2: Ranking (LightGBM) ---")
    reranker = Reranker()
    
    pos_samples = ratings[['user_id', 'item_id']].sample(2000).copy()
    pos_samples['target'] = 1
    
    all_items = item_features['item_id'].unique()
    neg_samples = pd.DataFrame({
        'user_id': np.random.choice(pos_samples['user_id'], 2000),
        'item_id': np.random.choice(all_items, 2000),
        'target': 0
    })
    
    train_df = pd.concat([pos_samples, neg_samples]).merge(item_features, on='item_id', how='left').fillna(0)
    reranker.train(train_df, 'target')

    print("\n--- ЭТАП 3: Валидация всей системы ---")
    results = {"recall": [], "mrr": [], "ndcg": []}
    
    model.eval()
    for uid, history, target in test_data[:200]:
        input_seq = quantizer.encode_session(history[-4:])
        if len(input_seq) < 4: input_seq = [0]*(4-len(input_seq)) + input_seq
        
        with torch.no_grad():
            logits = model(torch.tensor([input_seq]))
            top_codes = torch.topk(logits, 5).indices.numpy()[0]
        
        candidates = []
        for code in top_codes:
            candidates.extend(quantizer.decode_semantic_id(code))
        candidates = list(set(candidates))[:100] # Ограничиваем Stage 1

        final_recs = reranker.rank(candidates, uid, item_features, top_k=10)

        rec, mrr, ndcg = get_metrics([target], final_recs, k=10)
        results["recall"].append(rec)
        results["mrr"].append(mrr)
        results["ndcg"].append(ndcg)

    print(f"Средний Recall@10: {np.mean(results['recall']):.4f}")
    print(f"Средний MRR:       {np.mean(results['mrr']):.4f}")
    print(f"Средний NDCG@10:    {np.mean(results['ndcg']):.4f}")

if __name__ == "__main__":
    main()