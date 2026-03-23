## Generative Retrieval & Two-Stage Ranking (MovieLens 100k)

Scalable recommendation system prototype using **Deep Learning (Retrieval)** and **GBDT (Ranking)** with semantic quantization for optimized inference.

---

### System Architecture

#### 1. Semantic Quantization
* **Item Embeddings:** Generated via **Word2Vec (Skip-gram)** on user session sequences.
* **Codebook:** **K-Means** clustering of embeddings; centroids represent "semantic units" (similar item groups).
* **Indexing:** Each item is assigned a `semantic_id` to map the catalog into a discrete latent space.

#### 2. Stage 1: Generative Retrieval (PyTorch)
* **Model:** Autoregressive **GRU-based RNN**.
* **Task:** Predicts the next `semantic_id` based on session history.
* **Function:** Reduces search space while maintaining session context by operating on interest categories.

#### 3. Stage 2: Reranking (LightGBM)
* **Model:** **LGBMClassifier**.
* **Features:** Item popularity, average ratings, and static metadata.
* **Function:** Refines the RNN-generated top candidates using tabular business metrics.

---

### Validation & Results
**Temporal Split** evaluation (last user action held out for testing).

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Recall@10** | **0.1000** | Probability of target in top-10 |
| **MRR** | **0.0365** | Mean Reciprocal Rank |
| **NDCG@10** | **0.0508** | Position-weighted relevance |

*Performance is 16x higher than the Random Baseline ($\approx 0.006$).*

---

### Hyperparameters

**Stage 1: VQ & RNN**
* **Inventory:** ~1,600 items.
* **Codebook Size ($K$):** 64 clusters (~25x space compression).
* **Embeddings:** dim=32, window=10.
* **Sequence:** 5 (4 context + 1 target).
* **Architecture:** 1-layer GRU, hidden=64, 3 epochs.

**Stage 2: LightGBM**
* **Dataset:** 2,000 positive + 2,000 random negative samples.
* **Features:** `item_popularity`, `item_avg_rating`.
* **Model:** `LGBMClassifier` (50 estimators).