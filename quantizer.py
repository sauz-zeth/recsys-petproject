import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from multiprocessing import cpu_count

class ItemQuantizer:
    def __init__(self, n_clusters: int = 256, embedding_dim: int = 64):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.w2v_model = None
        self.kmeans_model = None
        self.item_to_semantic = {} # Mapping: item_id -> semantic_id
        self.semantic_to_items = {} # Mapping: semantic_id -> list[item_id]

    def train_embeddings(self, user_sessions: list[list[str]]):
        self.w2v_model = Word2Vec(
            sentences=user_sessions,
            sg=1,                 
            vector_size=self.embedding_dim,
            window=10,            
            min_count=2,          
            negative=20,          
            alpha=0.03,
            min_alpha=0.0007,
            workers=cpu_count() - 1
        )

        self.w2v_model.wv.fill_norms()

    def fit_kmeans(self):
        if self.w2v_model is None:
            raise ValueError("Word2Vec model is not trained. Call train_embeddings() first.")

        items = self.w2v_model.wv.index_to_key
        
        vectors = np.array([self.w2v_model.wv.get_vector(item, norm=True) for item in items])
        
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(vectors)
        
        self.item_to_semantic = {}
        self.semantic_to_items = {}
        
        for item, cluster_id in zip(items, cluster_labels):
            c_id = int(cluster_id) # преобразуем из numpy.int64 в обычный int
            
            self.item_to_semantic[item] = c_id
            
            if c_id not in self.semantic_to_items:
                self.semantic_to_items[c_id] = []
            self.semantic_to_items[c_id].append(item)
            
    def encode_session(self, session: list[str]) -> list[int]:
        return [self.item_to_semantic[item] for item in session if item in self.item_to_semantic]

    def decode_semantic_id(self, semantic_id: int) -> list[str]:
        return self.semantic_to_items.get(semantic_id, [])