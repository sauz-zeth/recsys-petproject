## Two-Stage Recommendation System (MovieLens 100k)

End-to-end recommendation pipeline combining semantic quantization, deep learning retrieval, and GBDT reranking — built to understand how production RecSys architectures work at each stage.

---

### Architecture

```
User Session History
       │
       ▼
┌─────────────────────────┐
│   Semantic Quantization │   Word2Vec (Skip-gram) → K-Means
│   1,682 items → 128 IDs │  ~13 items per cluster
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 1: GRU Retrieval │  Predicts top-10 semantic IDs
│  (PyTorch)              │  → ~200 candidate items
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Stage 2: LightGBM      │  Scores each candidate:
│  Reranking              │  LightGBM(popularity, avg_rating)
│                         │  × GRU cluster probability
└────────────┬────────────┘
             │
             ▼
        Top-10 Recommendations
```

**Semantic quantization** compresses the item catalog ~13× before passing it to the neural network. The GRU operates on 128 semantic IDs instead of 1,682 raw items, reducing inference-time search space. At ranking, the two signals are multiplied: LightGBM captures global item quality, GRU provides session-based personalization.

---

### Results

Evaluated on 943 users, **temporal split** (last interaction held out per user).

| Model | Recall@10 | MRR | NDCG@10 |
| :--- | :---: | :---: | :---: |
| Popularity Baseline | 0.0838 | 0.0314 | 0.0436 |
| **Two-Stage (ours)** | **0.1060** | **0.0351** | **0.0514** |
| **vs Baseline** | **+26.6%** | **+11.7%** | **+17.9%** |

---

### Hyperparameters

| Component | Parameter | Value |
| :--- | :--- | :--- |
| Word2Vec | architecture | Skip-gram, dim=64, window=10 |
| K-Means | codebook size *K* | 128 clusters |
| GRU | hidden / embed dim | 128 / 64 |
| GRU | training | 7 epochs, Adam lr=0.001, batch=256 |
| LightGBM | estimators | 50, trained on 2k pos + 2k neg |

---

### Getting Started

Requires [uv](https://docs.astral.sh/uv/):

```bash
pip install uv
```

```bash
git clone https://github.com/sauz-zeth/recsys-petproject.git
cd recsys-petproject
uv sync
uv run main.py
```

The MovieLens 100k dataset is downloaded automatically on first run.
