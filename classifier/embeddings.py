"""
Embeddings Module
Handles text embeddings and vector operations for similarity search
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Manages text embeddings and vector similarity operations.
    """
    
    def __init__(self, 
                 model_name: str = 'all-mpnet-base-v2',
                 cache_dir: str = './data/embeddings'):
        """
        Initialize embeddings manager.
        
        Args:
            model_name: Name of the sentence transformer model
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.indexed_texts = []
        self.metadata = []
        
        logger.info(f"EmbeddingsManager initialized with {model_name} "
                   f"(dim={self.embedding_dim})")
    
    def encode_texts(self, texts: List[str], 
                    batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def build_index(self, texts: List[str], 
                   metadata: List[Dict] = None,
                   save_path: str = None) -> faiss.IndexFlatL2:
        """
        Build FAISS index from texts.
        
        Args:
            texts: List of texts to index
            metadata: Optional metadata for each text
            save_path: Optional path to save the index
            
        Returns:
            FAISS index
        """
        if not texts:
            logger.warning("No texts provided for indexing")
            return None
        
        # Encode texts
        embeddings = self.encode_texts(texts)
        
        # Create FAISS index
        logger.info(f"Building FAISS index for {len(texts)} texts...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        
        # Store texts and metadata
        self.indexed_texts = texts
        self.metadata = metadata if metadata else [{} for _ in texts]
        
        # Save index if path provided
        if save_path:
            self.save_index(save_path)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query: str, 
              k: int = 10,
              threshold: float = None) -> List[Dict]:
        """
        Search for similar texts using the index.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of search results with text, score, and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No index available for search")
            return []
        
        # Encode query
        query_embedding = self.encode_texts([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + dist)
            
            # Apply threshold if specified
            if threshold and similarity < threshold:
                continue
            
            result = {
                'text': self.indexed_texts[idx],
                'similarity': float(similarity),
                'distance': float(dist),
                'rank': i + 1,
                'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
            }
            results.append(result)
        
        return results
    
    def search_batch(self, queries: List[str], 
                    k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple queries in batch.
        
        Args:
            queries: List of query texts
            k: Number of results per query
            
        Returns:
            List of search results for each query
        """
        if not queries:
            return []
        
        # Encode all queries
        query_embeddings = self.encode_texts(queries)
        
        # Search index
        distances, indices = self.index.search(query_embeddings, min(k, self.index.ntotal))
        
        # Prepare results for each query
        all_results = []
        for q_idx in range(len(queries)):
            results = []
            for i, (dist, idx) in enumerate(zip(distances[q_idx], indices[q_idx])):
                if idx == -1:
                    continue
                
                similarity = 1 / (1 + dist)
                result = {
                    'text': self.indexed_texts[idx],
                    'similarity': float(similarity),
                    'distance': float(dist),
                    'rank': i + 1,
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                }
                results.append(result)
            
            all_results.append(results)
        
        return all_results
    
    def add_to_index(self, texts: List[str], 
                    metadata: List[Dict] = None):
        """
        Add new texts to existing index.
        
        Args:
            texts: List of texts to add
            metadata: Optional metadata for each text
        """
        if not texts:
            return
        
        # Encode new texts
        embeddings = self.encode_texts(texts)
        
        # Add to index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.index.add(embeddings)
        
        # Update stored texts and metadata
        self.indexed_texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        
        logger.info(f"Added {len(texts)} texts to index (total: {self.index.ntotal})")
    
    def save_index(self, save_path: str):
        """
        Save index and associated data to disk.
        
        Args:
            save_path: Path to save the index
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path.with_suffix('.faiss')))
        
        # Save texts and metadata
        data = {
            'texts': self.indexed_texts,
            'metadata': self.metadata,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(save_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved index to {save_path}")
    
    def load_index(self, load_path: str):
        """
        Load index and associated data from disk.
        
        Args:
            load_path: Path to load the index from
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path.with_suffix('.faiss')))
        
        # Load texts and metadata
        with open(load_path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        self.indexed_texts = data['texts']
        self.metadata = data['metadata']
        
        # Verify model compatibility
        if data['model_name'] != self.model_name:
            logger.warning(f"Index was created with {data['model_name']}, "
                          f"but current model is {self.model_name}")
        
        logger.info(f"Loaded index from {load_path} ({self.index.ntotal} vectors)")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.encode_texts([text1, text2])
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def find_duplicates(self, texts: List[str], 
                       threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """
        Find duplicate or near-duplicate texts.
        
        Args:
            texts: List of texts to check
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of tuples (index1, index2, similarity)
        """
        if len(texts) < 2:
            return []
        
        embeddings = self.encode_texts(texts)
        duplicates = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity >= threshold:
                    duplicates.append((i, j, float(similarity)))
        
        return duplicates
    
    def cluster_texts(self, texts: List[str], 
                     n_clusters: int = None) -> Dict[int, List[int]]:
        """
        Cluster texts based on semantic similarity.
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary mapping cluster ID to text indices
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        if len(texts) < 2:
            return {0: list(range(len(texts)))}
        
        embeddings = self.encode_texts(texts)
        
        # Auto-determine optimal number of clusters if not specified
        if n_clusters is None:
            max_clusters = min(10, len(texts) - 1)
            best_score = -1
            best_n = 2
            
            for n in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n = n
            
            n_clusters = best_n
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Group indices by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        logger.info(f"Clustered {len(texts)} texts into {n_clusters} clusters")
        return clusters
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {'status': 'no_index'}
        
        return {
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'num_texts': len(self.indexed_texts),
            'has_metadata': len(self.metadata) > 0
        }
    
    def clear_index(self):
        """Clear the current index and associated data."""
        self.index = None
        self.indexed_texts = []
        self.metadata = []
        logger.info("Index cleared")
    
    def __repr__(self):
        n_vectors = self.index.ntotal if self.index else 0
        return f"EmbeddingsManager(model={self.model_name}, vectors={n_vectors})"