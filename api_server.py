#!/usr/bin/env python3
"""
FastAPI server to serve the trained Arvocap chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
from typing import List, Optional
import json
import os
from datetime import datetime

# Import your existing chatbot components
from chatbot_trainer import ChatbotTrainer, ChatbotInterface
from vector_database import ChatbotKnowledgeBase
from text_processor import TextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Arvocap Chatbot API",
    description="API for the Arvocap trained chatbot",
    version="1.0.0"
)

# Configure CORS to allow your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development with tunnels
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

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

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    timestamp: str
    sources: Optional[List[dict]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class KnowledgeSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class SetApiKeyRequest(BaseModel):
    provider: str
    api_key: str

class RetrainRequest(BaseModel):
    background_tasks: Optional[bool] = True
    processed_data_file: Optional[str] = 'processed_data.json'

# Global chatbot instance
chatbot_trainer = None
chatbot_interface = None
knowledge_base = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the server starts"""
    global chatbot_trainer, chatbot_interface, knowledge_base
    try:
        logger.info("🚀 Starting Arvocap Chatbot API...")
        
        # Initialize knowledge base
        knowledge_base = ChatbotKnowledgeBase()
        logger.info("✅ Knowledge base loaded")
        
        # Initialize chatbot trainer
        chatbot_trainer = ChatbotTrainer()
        logger.info("✅ Chatbot trainer initialized")
        
        # Initialize chatbot interface (this is what we'll use for chat)
        chatbot_interface = ChatbotInterface(use_openai=True)
        logger.info("✅ Chatbot interface initialized")
        
        logger.info("🎉 Arvocap Chatbot API is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        # Don't raise - allow server to start even if chatbot fails
        logger.warning("⚠️  Server starting without fully initialized chatbot")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Arvocap Chatbot API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "API server is working!", "timestamp": datetime.now().isoformat()}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    try:
        # Check if chatbot is ready
        if chatbot_trainer is None or knowledge_base is None or chatbot_interface is None:
            return HealthResponse(
                status="initializing",
                message="Chatbot is still initializing, please wait...",
                timestamp=datetime.now().isoformat()
            )
        
        # Check knowledge base
        try:
            collection_count = knowledge_base.get_collection_size()
            status_message = f"Chatbot ready with {collection_count} documents in knowledge base"
        except:
            collection_count = 0
            status_message = "Chatbot ready but knowledge base is empty"
        
        return HealthResponse(
            status="healthy",
            message=status_message,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            message=f"Service error: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        logger.info(f"🔍 Chat endpoint called with message: {chat_message.message}")
        
        if chatbot_interface is None:
            logger.error("❌ Chatbot interface not initialized")
            raise HTTPException(status_code=503, detail="Chatbot not initialized yet, please try again in a moment")
        
        logger.info(f"📝 Processing message: {chat_message.message}")
        
        # Get response from chatbot interface (same as CLI) - prioritize speed
        response = chatbot_interface.generate_response(chat_message.message)
        logger.info(f"🤖 Generated response length: {len(response)} chars")
        
        # Get relevant sources from knowledge base (reduced for speed)
        sources = []
        if knowledge_base is not None:
            try:
                relevant_docs = knowledge_base.search_similar_content(
                    chat_message.message, 
                    max_results=2  # Reduced from 3 to 2 for faster response
                )
                sources = [
                    {
                        "content": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"],  # Shorter snippets
                        "metadata": doc.get("metadata", {}),
                        "score": float(doc.get("score", 0.0))
                    }
                    for doc in relevant_docs
                ]
                logger.info(f"📚 Found {len(sources)} relevant sources")
            except Exception as e:
                logger.warning(f"Could not retrieve sources: {e}")
        
        # Generate conversation ID if not provided
        conversation_id = chat_message.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        chat_response = ChatResponse(
            response=response,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            sources=sources
        )
        
        logger.info(f"✅ Response sent successfully")
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

async def get_chatbot_response(message: str) -> str:
    """Get response from the trained chatbot"""
    try:
        # Use your trained chatbot to generate response
        response = chatbot_trainer.chat_with_bot(message)
        return response
    except Exception as e:
        logger.error(f"Chatbot response error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    try:
        if knowledge_base is None:
            return {
                "total_documents": 0,
                "last_updated": datetime.now().isoformat(),
                "status": "not_initialized"
            }
        
        try:
            doc_count = knowledge_base.get_collection_size()
        except:
            doc_count = 0
            
        stats = {
            "total_documents": doc_count,
            "last_updated": datetime.now().isoformat(),
            "status": "active"
        }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=VectorSearchResponse)
async def vector_search(request: VectorSearchRequest):
    """Vector search endpoint for Next.js integration"""
    try:
        if knowledge_base is None:
            # Return empty results if knowledge base not available
            return VectorSearchResponse(
                results=[],
                query=request.query,
                total_results=0
            )
        
        # Search the knowledge base
        raw_results = knowledge_base.search_similar_content(
            request.query, 
            max_results=request.max_results
        )
        
        # Filter by score threshold and format results
        filtered_results = []
        for doc in raw_results:
            score = float(doc.get("score", 0.0))
            if score >= request.score_threshold:
                filtered_results.append(VectorSearchResult(
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    score=score
                ))
        
        return VectorSearchResponse(
            results=filtered_results,
            query=request.query,
            total_results=len(filtered_results)
        )
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        # Return empty results on error to maintain API contract
        return VectorSearchResponse(
            results=[],
            query=request.query,
            total_results=0
        )

@app.post("/knowledge/search")
async def search_knowledge(request: KnowledgeSearchRequest):
    """Search the knowledge base"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        results = knowledge_base.search_similar_content(
            request.query, 
            max_results=request.max_results
        )
        return {"query": request.query, "results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def trigger_retraining(request: RetrainRequest):
    """Trigger retraining of the knowledge base"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        # Generate a job ID for tracking
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        background_tasks = request.background_tasks if request.background_tasks is not None else True
        processed_file = request.processed_data_file or 'processed_data.json'

        if background_tasks:
            # In a real implementation, you'd use FastAPI BackgroundTasks
            # For now, we'll just return success
            logger.info(f"Retraining job {job_id} queued")
            return {
                "success": True,
                "job_id": job_id,
                "message": "Retraining job queued",
                "status": "queued"
            }
        else:
            # Synchronous retraining (for testing)
            try:
                logger.info("Starting synchronous retraining...")
                report = knowledge_base.retrain(processed_file)
                if not report.get('success'):
                    raise Exception(report.get('error', 'Unknown error'))
                logger.info("Retraining completed")
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Retraining completed successfully",
                    "report": report,
                    "status": "completed"
                }
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                return {
                    "success": False,
                    "job_id": job_id,
                    "message": f"Retraining failed: {str(e)}",
                    "status": "failed"
                }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrain endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_detailed_status():
    """Get detailed status of all components"""
    status = {
        "api_server": "running",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "chatbot_trainer": chatbot_trainer is not None,
            "knowledge_base": knowledge_base is not None,
        }
    }
    
    if knowledge_base is not None:
        try:
            status["knowledge_base_size"] = knowledge_base.get_collection_size()
        except:
            status["knowledge_base_size"] = 0
    
    return status

@app.post("/admin/set_api_key")
async def set_api_key(request: SetApiKeyRequest):
    """Set provider API key at runtime. Currently supports provider='openai'"""
    try:
        provider = (request.provider or '').lower()
        key = request.api_key or ''
        if provider != 'openai':
            raise HTTPException(status_code=400, detail="Unsupported provider. Use 'openai'.")
        if not key:
            raise HTTPException(status_code=400, detail="API key is required")

        # Update in both trainer and interface if available
        if chatbot_trainer is not None:
            try:
                chatbot_trainer.update_openai_api_key(key)
            except Exception as e:
                logger.warning(f"Failed to set key in trainer: {e}")
        if chatbot_interface is not None:
            try:
                chatbot_interface.set_openai_api_key(key)
            except Exception as e:
                logger.warning(f"Failed to set key in interface: {e}")

        return {"success": True, "provider": provider}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Arvocap Chatbot API Server...")
    print("📡 Server will be available at: http://127.0.0.1:8000")
    print("📖 API Documentation at: http://127.0.0.1:8000/docs")
    print("🔧 Health check at: http://127.0.0.1:8000/health")
    print("⚡ Optimized for fast responses")
    
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False,  # Disable access logs for better performance
        workers=1          # Single worker for consistency
    )
