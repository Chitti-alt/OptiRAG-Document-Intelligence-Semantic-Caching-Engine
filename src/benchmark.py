import time
from main import ask_rag_system

def run_benchmark():
    test_queries = [
        "What are the primary assumptions in lifting line theory?",
        "Define what no swirl at exit means in an axial turbine.",
        "How are turbomachinery velocity triangles configured?",
        "Can you explain the meaning of having no swirl at the exit of an axial turbine?",
        "Should turbomachinery velocity triangles be drawn contiguous with a common base?",
        "What does lifting line theory assume about the vortex sheet?"
    ]
    
    total_time_api = 0
    total_time_cache = 0
    cache_hits = 0
    
    print("🚀 Starting Automated RAG Benchmark...\n")
    
    for query in test_queries:
        start = time.time()
        
        answer, latency, hit_cache = ask_rag_system(query) 
        
        if hit_cache:
            cache_hits += 1
            total_time_cache += latency
        else:
            total_time_api += latency
            
        print(f"Q: {query[:50]}... | Cache Hit: {hit_cache} | Time: {latency:.2f}ms")
        time.sleep(1) 
        
    avg_api_latency = total_time_api / (len(test_queries) - cache_hits) if (len(test_queries) - cache_hits) > 0 else 0
    avg_cache_latency = total_time_cache / cache_hits if cache_hits > 0 else 0
    latency_reduction = ((avg_api_latency - avg_cache_latency) / avg_api_latency) * 100 if avg_api_latency > 0 else 0
    
    print("\n" + "="*40)
    print("📊 BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Queries: {len(test_queries)}")
    print(f"Cache Hit Rate: {(cache_hits / len(test_queries)) * 100:.1f}%")
    print(f"Average API Latency: {avg_api_latency:.2f} ms")
    print(f"Average Cache Latency: {avg_cache_latency:.2f} ms")
    print(f"Speed Optimization: {latency_reduction:.1f}% faster")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
