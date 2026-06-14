import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    def __init__(self, similarity_threshold=0.85):
        # Uses a fast, free local model to convert text to math vectors
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_queries = []
        self.cache_answers = []
        self.threshold = similarity_threshold

    def check_cache(self, user_query):
        start_time = time.time()
        query_vector = self.encoder.encode([user_query])

        if not self.cache_queries:
            return None, 0

        # Check mathematical similarity against all previously asked questions
        similarities = cosine_similarity(query_vector, self.cache_queries)[0]
        best_match_idx = np.argmax(similarities)

        if similarities[best_match_idx] >= self.threshold:
            latency = (time.time() - start_time) * 1000
            print(f"⚡ CACHE HIT! Similarity: {similarities[best_match_idx]:.2f}")
            return self.cache_answers[best_match_idx], latency
        
        return None, 0

    def add_to_cache(self, user_query, llm_answer):
        query_vector = self.encoder.encode([user_query])[0]
        self.cache_queries.append(query_vector)
        self.cache_answers.append(llm_answer)
