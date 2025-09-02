#!/usr/bin/env python3
"""
Unified Arvocap Chatbot System
Combines web scraping, PDF processing, and chatbot serving in one application
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import argparse
from concurrent.futures import ThreadPoolExecutor

# FastAPI and web components
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import existing components
try:
    from web_scraper import WebScraper
    from pdf_processor import PDFProcessor  
    from text_processor import TextProcessor
    from vector_database import ChatbotKnowledgeBase
    from chatbot_trainer import ChatbotInterface
    from config import DATA_PATH, VECTOR_DB_PATH
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all modules are available")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Arvocap Chatbot System",
    description="Complete chatbot system with web scraping, PDF processing, and AI chat",
    version="2.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class WebScrapingRequest(BaseModel):
    urls: List[str]
    crawl_depth: Optional[int] = 1
    max_pages: Optional[int] = 10

class KnowledgeBaseStatus(BaseModel):
    total_documents: int
    last_updated: str
    sources: Dict[str, int]

class TrainingRequest(BaseModel):
    clear_existing: Optional[bool] = False
    include_web: Optional[bool] = True
    include_pdfs: Optional[bool] = True
    urls: Optional[List[str]] = None

# Global components
class UnifiedChatbotSystem:
    """
    Unified system that handles all chatbot operations
    """
    
    def __init__(self):
        """Initialize all components"""
        self.data_dir = Path(DATA_PATH)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.web_scraper = WebScraper()
        self.pdf_processor = PDFProcessor(use_ocr=True)
        self.text_processor = TextProcessor()
        self.knowledge_base = ChatbotKnowledgeBase()
        self.chatbot = ChatbotInterface(use_openai=True)
        
        # Training status
        self.training_in_progress = False
        self.last_training_time = None
        
        logger.info("Unified Chatbot System initialized")
    
    async def scrape_websites(self, urls: List[str], max_pages: int = 10) -> Dict[str, Any]:
        """
        Scrape websites and extract content
        """
        logger.info(f"Starting web scraping for {len(urls)} URLs")
        
        scraped_data = {}
        total_pages = 0
        
        for url in urls:
            if total_pages >= max_pages:
                break
                
            try:
                logger.info(f"Scraping: {url}")
                result = self.web_scraper.scrape_website(url)
                
                if result:
                    scraped_data[url] = result
                    total_pages += 1
                    logger.info(f"✅ Scraped: {url}")
                else:
                    logger.warning(f"⚠️ Failed to scrape: {url}")
                    
            except Exception as e:
                logger.error(f"❌ Error scraping {url}: {e}")
        
        return {
            "scraped_data": scraped_data,
            "total_pages": total_pages,
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_pdfs(self, pdf_directory: str = None) -> Dict[str, Any]:
        """
        Process PDF files and extract content
        """
        if not pdf_directory:
            pdf_directory = self.data_dir / "pdfs"
        
        logger.info(f"Processing PDFs from: {pdf_directory}")
        
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_directory}")
            return {"processed_pdfs": [], "total_chunks": 0}
        
        # Get all PDF files
        pdf_files = list(pdf_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        processed_pdfs = []
        total_chunks = 0
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                result = self.pdf_processor.process_pdf_with_ocr(str(pdf_file))
                
                if result:
                    processed_pdfs.append(result)
                    chunks = result.get('stats', {}).get('total_chunks', 0)
                    total_chunks += chunks
                    logger.info(f"✅ Processed {pdf_file.name}: {chunks} chunks")
                else:
                    logger.warning(f"⚠️ Failed to process: {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {pdf_file.name}: {e}")
        
        return {
            "processed_pdfs": processed_pdfs,
            "total_chunks": total_chunks,
            "total_files": len(processed_pdfs),
            "timestamp": datetime.now().isoformat()
        }
    
    async def update_knowledge_base(self, scraped_data: Dict = None, pdf_data: Dict = None) -> bool:
        """
        Update knowledge base with new content
        """
        logger.info("Updating knowledge base...")
        
        try:
            added_docs = 0
            
            # Process scraped web data
            if scraped_data:
                for url, content in scraped_data.get("scraped_data", {}).items():
                    if isinstance(content, dict) and content.get('text'):
                        # Create chunks from web content
                        chunks = self.text_processor.chunk_text(content['text'])
                        
                        for i, chunk in enumerate(chunks):
                            metadata = {
                                "source": url,
                                "type": "web",
                                "chunk_id": f"{url}_chunk_{i}",
                                "scraped_at": content.get('timestamp')
                            }
                            self.knowledge_base.add_document(chunk, metadata)
                            added_docs += 1
            
            # Process PDF data
            if pdf_data:
                for pdf in pdf_data.get("processed_pdfs", []):
                    content = pdf.get('content', {})
                    metadata = pdf.get('metadata', {})
                    chunks = content.get('chunks', [])
                    
                    for i, chunk in enumerate(chunks):
                        if chunk and len(chunk.strip()) > 20:
                            chunk_metadata = {
                                "source": metadata.get('filename', 'unknown'),
                                "type": "pdf",
                                "chunk_id": f"{metadata.get('filename', 'unknown')}_chunk_{i}",
                                "processing_method": pdf.get('stats', {}).get('extraction_method', 'unknown'),
                                "processed_at": datetime.now().isoformat()
                            }
                            self.knowledge_base.add_document(chunk, chunk_metadata)
                            added_docs += 1
            
            logger.info(f"✅ Added {added_docs} documents to knowledge base")
            self.last_training_time = datetime.now().isoformat()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update knowledge base: {e}")
            return False
    
    async def full_training_pipeline(self, 
                                   urls: List[str] = None, 
                                   include_web: bool = True,
                                   include_pdfs: bool = True,
                                   clear_existing: bool = False) -> Dict[str, Any]:
        """
        Run complete training pipeline
        """
        if self.training_in_progress:
            raise HTTPException(status_code=409, detail="Training already in progress")
        
        self.training_in_progress = True
        
        try:
            logger.info("🚀 Starting unified training pipeline")
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("🗑️ Clearing existing knowledge base")
                self.knowledge_base.clear_database()
            
            results = {
                "started_at": datetime.now().isoformat(),
                "web_scraping": None,
                "pdf_processing": None,
                "knowledge_base_update": False,
                "final_stats": {}
            }
            
            # Web scraping
            scraped_data = None
            if include_web and urls:
                logger.info("🌐 Starting web scraping...")
                scraped_data = await self.scrape_websites(urls)
                results["web_scraping"] = {
                    "urls_processed": len(scraped_data.get("scraped_data", {})),
                    "total_pages": scraped_data.get("total_pages", 0)
                }
            
            # PDF processing  
            pdf_data = None
            if include_pdfs:
                logger.info("📄 Starting PDF processing...")
                pdf_data = await self.process_pdfs()
                results["pdf_processing"] = {
                    "files_processed": pdf_data.get("total_files", 0),
                    "total_chunks": pdf_data.get("total_chunks", 0)
                }
            
            # Update knowledge base
            if scraped_data or pdf_data:
                logger.info("🔍 Updating knowledge base...")
                success = await self.update_knowledge_base(scraped_data, pdf_data)
                results["knowledge_base_update"] = success
            
            # Final statistics
            kb_size = self.knowledge_base.get_collection_size()
            results["final_stats"] = {
                "total_documents_in_kb": kb_size,
                "training_completed_at": datetime.now().isoformat()
            }
            
            logger.info("🎉 Training pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
        
        finally:
            self.training_in_progress = False
    
    def get_knowledge_base_status(self) -> KnowledgeBaseStatus:
        """Get current knowledge base status"""
        kb_size = self.knowledge_base.get_collection_size()
        
        # Try to get source breakdown (this would need to be implemented)
        sources = {"web": 0, "pdf": 0, "manual": 0}
        
        return KnowledgeBaseStatus(
            total_documents=kb_size,
            last_updated=self.last_training_time or "Never",
            sources=sources
        )

# Initialize the unified system
unified_system = UnifiedChatbotSystem()

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system info"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified Arvocap Chatbot System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #20405A; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status { color: #28a745; }
        </style>
    </head>
    <body>
        <h1 class="header">🤖 Unified Arvocap Chatbot System</h1>
        <p class="status">✅ System Online</p>
        
        <h2>Available Endpoints:</h2>
        <div class="endpoint"><strong>POST /chat</strong> - Chat with the AI</div>
        <div class="endpoint"><strong>POST /scrape</strong> - Scrape websites</div>
        <div class="endpoint"><strong>POST /train</strong> - Run training pipeline</div>
        <div class="endpoint"><strong>GET /status</strong> - Get knowledge base status</div>
        <div class="endpoint"><strong>GET /docs</strong> - API documentation</div>
        
        <h2>Features:</h2>
        <ul>
            <li>🌐 Web scraping and content extraction</li>
            <li>📄 PDF processing with OCR support</li>
            <li>🤖 AI-powered chat responses</li>
            <li>🔍 Vector-based knowledge search</li>
            <li>📊 Real-time training and updates</li>
        </ul>
        
        <p><a href="/docs">📚 View Full API Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    """Chat with the trained AI"""
    try:
        logger.info(f"Chat request: {message.message[:100]}...")
        
        response = unified_system.chatbot.generate_response(message.message)
        
        return {
            "response": response,
            "conversation_id": message.conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/scrape")
async def scrape_websites(request: WebScrapingRequest, background_tasks: BackgroundTasks):
    """Scrape websites and optionally update knowledge base"""
    try:
        logger.info(f"Web scraping request for {len(request.urls)} URLs")
        
        # Start scraping
        result = await unified_system.scrape_websites(
            urls=request.urls,
            max_pages=request.max_pages
        )
        
        return {
            "status": "completed",
            "urls_processed": len(result.get("scraped_data", {})),
            "total_pages": result.get("total_pages", 0),
            "timestamp": result.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/train")
async def train_system(request: TrainingRequest):
    """Run the complete training pipeline"""
    try:
        logger.info("Training request received")
        
        result = await unified_system.full_training_pipeline(
            urls=request.urls,
            include_web=request.include_web,
            include_pdfs=request.include_pdfs,
            clear_existing=request.clear_existing
        )
        
        return {
            "status": "completed",
            "results": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system and knowledge base status"""
    try:
        kb_status = unified_system.get_knowledge_base_status()
        
        return {
            "system_status": "online",
            "training_in_progress": unified_system.training_in_progress,
            "knowledge_base": kb_status.dict(),
            "components": {
                "web_scraper": "available",
                "pdf_processor": "available", 
                "ocr": unified_system.pdf_processor.ocr_available,
                "chatbot": "available"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a single PDF file"""
    try:
        # Save uploaded file
        upload_dir = Path(DATA_PATH) / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        result = unified_system.pdf_processor.process_pdf_with_ocr(str(file_path))
        
        if result:
            # Update knowledge base
            await unified_system.update_knowledge_base(pdf_data={"processed_pdfs": [result]})
            
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_extracted": result.get('stats', {}).get('total_chunks', 0),
                "processing_method": result.get('stats', {}).get('extraction_method', 'unknown')
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to process PDF")
            
    except Exception as e:
        logger.error(f"PDF upload error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

# CLI Interface
def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(description="Unified Arvocap Chatbot System")
    
    parser.add_argument('--mode', choices=['server', 'train', 'scrape', 'status'], 
                       default='server', help='Operation mode')
    
    # Server options
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Training options
    parser.add_argument('--urls', nargs='+', help='URLs to scrape')
    parser.add_argument('--no-web', action='store_true', help='Skip web scraping')
    parser.add_argument('--no-pdfs', action='store_true', help='Skip PDF processing')
    parser.add_argument('--clear', action='store_true', help='Clear existing data')
    
    return parser

async def run_cli_mode(args):
    """Run CLI commands"""
    system = UnifiedChatbotSystem()
    
    if args.mode == 'status':
        status = system.get_knowledge_base_status()
        print(f"📊 Knowledge Base Status:")
        print(f"   Total documents: {status.total_documents}")
        print(f"   Last updated: {status.last_updated}")
        print(f"   Sources: {status.sources}")
        
    elif args.mode == 'scrape':
        if not args.urls:
            print("❌ No URLs provided for scraping")
            return
        
        print(f"🌐 Scraping {len(args.urls)} URLs...")
        result = await system.scrape_websites(args.urls)
        print(f"✅ Scraped {result['total_pages']} pages")
        
    elif args.mode == 'train':
        print("🚀 Starting training pipeline...")
        
        result = await system.full_training_pipeline(
            urls=args.urls,
            include_web=not args.no_web,
            include_pdfs=not args.no_pdfs,
            clear_existing=args.clear
        )
        
        print("✅ Training completed!")
        print(f"   Final KB size: {result['final_stats']['total_documents_in_kb']}")

def main():
    """Main entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    if args.mode == 'server':
        import uvicorn
        print("🚀 Starting Unified Arvocap Chatbot System")
        print(f"🌐 Server will be available at: http://{args.host}:{args.port}")
        print("📚 API docs at: http://{args.host}:{args.port}/docs")
        
        uvicorn.run(
            "unified_chatbot_system:app",
            host=args.host,
            port=args.port,
            reload=False
        )
    else:
        # Run CLI mode
        asyncio.run(run_cli_mode(args))

if __name__ == "__main__":
    main()
