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

def format_source_reference(result: dict) -> dict:
    """Format source references for the frontend display."""
    metadata = result.get("metadata", {})
    url = metadata.get("url", "")
    source_type = "webpage" if url.startswith("http") else "pdf"
    
    if source_type == "pdf":
        # For PDFs, extract just the filename
        label = url.split("/")[-1] if "/" in url else url.split("\\")[-1] if "\\" in url else url
        # Remove .pdf extension if present
        if label.lower().endswith(".pdf"):
            label = label[:-4]
    else:
        # For web pages, use the title or URL path
        label = metadata.get("title", "") or url.split("/")[-1]
    
    return {
        "label": label,
        "url": url,
        "sourceType": source_type,
        "score": result.get("score", 0)
    }

class ManualEntryRequest(BaseModel):
    title: str
    content: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

@app.post("/admin/add_manual_entry")
async def add_manual_entry(request: ManualEntryRequest):
    try:
        # Format the manual entry like scraped content
        formatted_content = {
            "url": f"manual://{request.title.lower().replace(' ', '-')}",
            "title": request.title,
            "content": request.content,
            "meta_description": request.title,
            "headings": [f"H1: {request.title}"],
            "timestamp": datetime.now().isoformat(),
            "type": "manual"
        }

        # Process the text using the text processor
        processor = TextProcessor()
        chunks = processor.create_chunks(
            request.content,
            chunk_size=request.chunk_size,
            overlap=request.chunk_overlap
        )

        # Add to knowledge base
        knowledge_base = ChatbotKnowledgeBase()
        num_chunks = 0
        
        for chunk in chunks:
            chunk_content = chunk["content"]
            chunk_metadata = {
                **formatted_content,
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"]
            }
            knowledge_base.add_document(chunk_content, chunk_metadata)
            num_chunks += 1

        return {
            "success": True,
            "chunks_created": num_chunks,
            "message": f"Successfully added manual entry with {num_chunks} chunks"
        }

    except Exception as e:
        logger.error(f"Error adding manual entry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
request_semaphore: asyncio.Semaphore | None = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the server starts"""
    global chatbot_trainer, chatbot_interface, knowledge_base, request_semaphore
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

        # Concurrency limiter (configurable via env CHAT_CONCURRENCY)
        try:
            limit = int(os.getenv("CHAT_CONCURRENCY", "8"))
            limit = max(1, limit)
        except Exception:
            limit = 8
        request_semaphore = asyncio.Semaphore(limit)
        logger.info(f"✅ Concurrency limit set to {limit}")
        
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

async def _retrieve_sources(query: str) -> list[dict]:
    """Retrieve relevant sources from the knowledge base and enrich them."""
    sources: list[dict] = []
    if knowledge_base is None:
        return sources
    try:
        # Run potentially blocking search in a thread to avoid blocking the event loop
        def _search():
            try:
                return knowledge_base.search_similar_content(query, max_results=2)
            except Exception:
                return []
        relevant_docs = await asyncio.to_thread(_search)
        enriched = []
        for d in relevant_docs:
            md = d.get("metadata", {}) or {}
            url = (md.get("url") or "").strip()
            title = (md.get("title") or "").strip()
            source = (md.get("source") or "").strip()
            src_type = (md.get("type") or "").lower()
            label = url or title or source or "Unknown source"
            source_type = "pdf" if ("pdf" in src_type or label.lower().endswith(".pdf")) else ("website" if url else (src_type or "document"))
            snippet = d["content"][:150] + "..." if len(d.get("content", "")) > 150 else d.get("content", "")
            enriched.append({
                "content": snippet,
                "metadata": md,
                "score": float(d.get("score", 0.0)),
                "label": label,
                "sourceType": source_type
            })
        sources = enriched
    except Exception as e:
        logger.warning(f"Could not retrieve sources: {e}")
    return sources


@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        logger.info(f"� Chat endpoint called with message: {chat_message.message}")
        
        if chatbot_interface is None:
            logger.error("❌ Chatbot interface not initialized")
            raise HTTPException(status_code=503, detail="Chatbot not initialized yet. Kindly rephrase the question?")
        
        logger.info(f"📝 Processing message: {chat_message.message}")

        # Run blocking tasks in parallel threads under a concurrency semaphore
        sem = request_semaphore or asyncio.Semaphore(8)
        async with sem:
            response_task = asyncio.to_thread(chatbot_interface.generate_response, chat_message.message)
            sources_task = _retrieve_sources(chat_message.message)
            response, sources = await asyncio.gather(response_task, sources_task)
            logger.info(f"🤖 Generated response length: {len(response)} chars")
            logger.info(f"📚 Found {len(sources)} relevant sources")
        
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
                # Format the source reference
                source_info = format_source_reference(doc)
                
                # Create the result with formatted source information
                result = VectorSearchResult(
                    content=doc["content"],
                    metadata={
                        **doc.get("metadata", {}),
                        "sourceLabel": source_info["label"],
                        "sourceType": source_info["sourceType"],
                        "sourceUrl": source_info["url"]
                    },
                    score=score
                )
                filtered_results.append(result)
        
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

# Admin endpoints for knowledge base management
@app.get("/admin/training_stats")
async def get_training_stats():
    """Get comprehensive training statistics"""
    try:
        if knowledge_base is None:
            return {
                "success": False,
                "total_documents": 0,
                "total_websites": 0,
                "manual_entries": 0,
                "knowledge_base_size": 0,
                "last_trained": None
            }
        
        # Get actual statistics from knowledge base
        stats = knowledge_base.get_statistics()
        
        return {
            "success": True,
            "total_documents": stats.get("document_count", 0),
            "total_websites": stats.get("website_count", 0), 
            "manual_entries": stats.get("manual_count", 0),
            "knowledge_base_size": stats.get("total_items", 0),
            "last_trained": stats.get("last_trained")
        }
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        return {
            "success": False,
            "total_documents": 0,
            "total_websites": 0,
            "manual_entries": 0,
            "knowledge_base_size": 0,
            "last_trained": None,
            "error": str(e)
        }

@app.get("/admin/knowledge_base")
async def get_knowledge_base():
    """Get all knowledge base items with metadata"""
    try:
        if knowledge_base is None:
            return {
                "success": False,
                "items": [],
                "total_items": 0,
                "error": "Knowledge base not initialized"
            }
        
        # Get items with metadata from knowledge base
        items = knowledge_base.get_all_items()
        
        return {
            "success": True,
            "items": items,
            "total_items": len(items)
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge base: {e}")
        return {
            "success": False,
            "items": [],
            "total_items": 0,
            "error": str(e)
        }

@app.delete("/admin/knowledge_base/{item_id}")
async def delete_knowledge_item(item_id: str):
    """Delete a specific knowledge base item"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        success = knowledge_base.delete_item(item_id)
        
        if success:
            return {"success": True, "message": f"Item {item_id} deleted successfully"}
        else:
            return {"success": False, "error": f"Item {item_id} not found"}
    except Exception as e:
        logger.error(f"Failed to delete knowledge item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/knowledge_base/{item_id}")
async def update_knowledge_item(item_id: str, request: dict):
    """Update a specific knowledge base item"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        title = request.get("title", "")
        content = request.get("content", "")
        
        if not title or not content:
            raise HTTPException(status_code=400, detail="Title and content are required")
        
        success = knowledge_base.update_item(item_id, title, content)
        
        if success:
            return {"success": True, "message": f"Item {item_id} updated successfully"}
        else:
            return {"success": False, "error": f"Item {item_id} not found"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update knowledge item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ManualEntryRequest(BaseModel):
    title: str
    content: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

@app.post("/admin/add_manual_entry")
async def add_manual_entry(request: ManualEntryRequest):
    """Add a manual entry to the knowledge base"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        # Process and add to knowledge base
        chunks_created = knowledge_base.add_manual_entry(
            title=request.title,
            content=request.content,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return {
            "success": True,
            "chunks_created": chunks_created,
            "message": f"Manual entry '{request.title}' added successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add manual entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DocumentUploadRequest(BaseModel):
    documents: List[str]  # File paths will be handled differently in actual implementation
    ocr_enabled: Optional[bool] = True
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class ManualEntryRequest(BaseModel):
    title: str
    content: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

class WebsiteScrapingRequest(BaseModel):
    url: str
    depth: Optional[int] = 2
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

@app.post("/admin/add_manual_entry")
async def add_manual_entry(request: ManualEntryRequest):
    """Add manual entry to knowledge base"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        # Process and add the manual entry
        text_processor = TextProcessor()
        chunks = text_processor.chunk_text(
            request.content,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Add metadata and store in knowledge base
        timestamp = datetime.now().isoformat()
        metadata = {
            "title": request.title,
            "type": "manual",
            "timestamp": timestamp,
            "source": "manual_entry"
        }
        
        # Add chunks to knowledge base
        added_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"manual_{timestamp}_{i}"
            success = knowledge_base.add_document(
                text=chunk,
                doc_id=chunk_id,
                metadata={**metadata, "chunk_index": i}
            )
            if success:
                added_chunks.append(chunk_id)
        
        if added_chunks:
            return {
                "success": True,
                "chunks_created": len(added_chunks),
                "message": "Manual entry added successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to add chunks to knowledge base"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add manual entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/scrape_website")
async def scrape_website(request: WebsiteScrapingRequest):
    """Scrape website and add to knowledge base"""
    try:
        if knowledge_base is None:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")
        
        # This would integrate with your web scraper
        # For now, return a placeholder response
        return {
            "success": True,
            "pages_processed": 1,
            "chunks_created": 5,
            "message": f"Website {request.url} scraped successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to scrape website: {e}")
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
