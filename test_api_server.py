#!/usr/bin/env python3
"""
Simplified FastAPI server for testing the integration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime

app = FastAPI(
    title="ArvoCap Chatbot API (Test)",
    description="Test API for the ArvoCap chatbot integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VectorSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    score_threshold: Optional[float] = 0.6

class VectorSearchResult(BaseModel):
    content: str
    metadata: dict
    score: float

class VectorSearchResponse(BaseModel):
    results: List[VectorSearchResult]
    query: str
    total_results: int

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

# Mock data for testing
mock_knowledge_base = [
    {
        "content": "ArvoCap Asset Managers Ltd offers comprehensive investment solutions including Money Market Funds with average returns of 16.5% and Thamani Equity Fund for aggressive growth strategies.",
        "metadata": {"category": "investment_products", "source": "company_info"},
        "score": 0.95
    },
    {
        "content": "Our Money Market Fund invests in government securities (18.39%), term and call deposits (80.16%), and maintains cash reserves (1.45%) for optimal liquidity and returns.",
        "metadata": {"category": "money_market", "source": "fund_allocation"},
        "score": 0.88
    },
    {
        "content": "The Thamani Equity Fund targets capital growth through strategic investment in NSE 25 Index stocks with active equity allocation and tactical positioning including derivatives hedging.",
        "metadata": {"category": "equity_fund", "source": "fund_strategy"},
        "score": 0.82
    }
]

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="ArvoCap Test API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="Test API ready with mock knowledge base",
        timestamp=datetime.now().isoformat()
    )

@app.post("/search", response_model=VectorSearchResponse)
async def vector_search(request: VectorSearchRequest):
    """Vector search endpoint for Next.js integration"""
    try:
        query_lower = request.query.lower()
        
        # Simple keyword matching for testing
        relevant_results = []
        for item in mock_knowledge_base:
            content_lower = item["content"].lower()
            
            # Basic relevance scoring
            score = 0.0
            query_words = query_lower.split()
            
            for word in query_words:
                if word in content_lower:
                    score += 0.2
                    
            # Boost for specific keywords
            if any(keyword in query_lower for keyword in ["money market", "fund", "investment", "return"]):
                if "money market" in content_lower:
                    score += 0.3
            
            if any(keyword in query_lower for keyword in ["thamani", "equity", "growth"]):
                if "thamani" in content_lower or "equity" in content_lower:
                    score += 0.3
                    
            if score >= request.score_threshold:
                relevant_results.append(VectorSearchResult(
                    content=item["content"],
                    metadata=item["metadata"],
                    score=score
                ))
        
        # Sort by score and limit results
        relevant_results.sort(key=lambda x: x.score, reverse=True)
        relevant_results = relevant_results[:request.max_results]
        
        return VectorSearchResponse(
            results=relevant_results,
            query=request.query,
            total_results=len(relevant_results)
        )
        
    except Exception as e:
        print(f"Search error: {e}")
        return VectorSearchResponse(
            results=[],
            query=request.query,
            total_results=0
        )

@app.post("/retrain")
async def trigger_retraining():
    """Mock retraining endpoint"""
    job_id = f"test_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "success": True,
        "job_id": job_id,
        "message": "Mock retraining completed",
        "status": "completed"
    }

@app.get("/status")
async def get_status():
    """Get detailed status"""
    return {
        "api_server": "running",
        "timestamp": datetime.now().isoformat(),
        "mode": "test",
        "knowledge_base_size": len(mock_knowledge_base)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting ArvoCap Test API Server...")
    print("📡 Server will be available at: http://127.0.0.1:8000")
    print("📖 API Documentation at: http://127.0.0.1:8000/docs")
    print("🔧 Health check at: http://127.0.0.1:8000/health")
    
    uvicorn.run(
        "test_api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
