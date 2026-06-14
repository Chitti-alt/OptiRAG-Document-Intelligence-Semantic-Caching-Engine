import os
import time
import chromadb
from google import genai
from cache import SemanticCache

client = genai.Client()

# Connect to the local Vector Database populated by ingest.py
db_client = chromadb.PersistentClient(path="./vector_db")
collection = db_client.get_or_create_collection(name="engineering_docs")

cache = SemanticCache(similarity_threshold=0.85)

def ask_rag_system(query: str):
    # 1. Check Cache
    cached_answer, latency = cache.check_cache(query)
    if cached_answer:
        return cached_answer, latency, True
        
    start_time = time.time()
    
    # 2. Retrieve Document Context
    results = collection.query(query_texts=[query], n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        return "No relevant context found in the database.", 0, False
        
    context = "\n".join(results['documents'][0])
    
    # 3. Prompt API
    prompt = f"""
    You are an expert technical engineering assistant. Answer the user's question based STRICTLY on the provided context. 
    If the context does not contain the answer, explicitly state "I do not have enough information."
    
    Context:
    {context}
    
    Question:
    {query}
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    
    answer = response.text.strip()
    latency = (time.time() - start_time) * 1000
    
    # 4. Save to Cache
    cache.add_to_cache(query, answer)
    
    return answer, latency, False
