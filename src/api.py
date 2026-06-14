from fastapi import FastAPI
from pydantic import BaseModel
from main import ask_rag_system

app = FastAPI(title="OptiRAG API", description="Layout-Aware RAG with Semantic Caching")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    latency_ms: float
    cache_hit: bool

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    # Route the web request to the core RAG engine
    answer, latency, cache_hit = ask_rag_system(request.question)
    
    return QueryResponse(
        answer=answer,
        latency_ms=round(latency, 2),
        cache_hit=cache_hit
    )
