#!/usr/bin/env python3
"""
Arvocap Chatbot Training Pipeline
Comprehensive training script that processes multiple data sources including PDFs, web scraped content, and manual data
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Local imports
from web_scraper import WebScraper
from text_processor import TextProcessor
from pdf_processor import PDFProcessor
from vector_database import VectorDatabase, ChatbotKnowledgeBase
from chatbot_trainer import ChatbotTrainer
from config import DATA_PATH, VECTOR_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArvocapTrainingPipeline:
    """
    Main training pipeline that orchestrates data collection, processing, and model training
    """
    
    def __init__(self, data_dir: str = DATA_PATH):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "raw_data").mkdir(exist_ok=True)
        (self.data_dir / "pdfs").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        
        # Initialize processors with FAISS
        self.web_scraper = WebScraper()
        self.text_processor = TextProcessor()
        self.pdf_processor = PDFProcessor()
        self.vector_db = VectorDatabase("vector_db_faiss")  # Use FAISS
        self.knowledge_base = ChatbotKnowledgeBase()
        self.trainer = ChatbotTrainer()
        
        logger.info("Training pipeline initialized")
    
    def collect_web_data(self, urls: List[str] = None) -> Dict[str, Any]:
        """
        Step 1: Collect data from web sources
        """
        logger.info("🌐 Step 1: Collecting web data...")
        
        if not urls:
            # Default Arvocap-related URLs
            urls = [
                "https://www.arvocap.com",
                # Add more relevant URLs here
            ]
        
        scraped_data = {}
        for url in urls:
            try:
                data = self.web_scraper.scrape_website(url)
                if data:
                    scraped_data[url] = data
                    logger.info(f"✅ Scraped data from {url}")
                else:
                    logger.warning(f"⚠️ No data scraped from {url}")
            except Exception as e:
                logger.error(f"❌ Failed to scrape {url}: {e}")
        
        # Save scraped data
        scraped_file = self.data_dir / "scraped_data.json"
        with open(scraped_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraped data saved to {scraped_file}")
        return scraped_data
    
    def collect_pdf_data(self, pdf_directory: str = None) -> List[Dict[str, Any]]:
        """
        Step 2: Process PDF documents
        """
        logger.info("📄 Step 2: Processing PDF documents...")
        
        if not pdf_directory:
            pdf_directory = self.data_dir / "pdfs"
        
        pdf_path = Path(pdf_directory)
        if not pdf_path.exists():
            logger.warning(f"PDF directory not found: {pdf_directory}")
            logger.info("Creating PDF directory. Place your PDF files there and run again.")
            pdf_path.mkdir(parents=True, exist_ok=True)
            return []
        
        # Process all PDFs in the directory
        processed_pdfs = self.pdf_processor.process_pdf_directory(str(pdf_path))
        
        if processed_pdfs:
            # Save processed PDF data
            pdf_output_file = self.data_dir / "processed" / "processed_pdfs.json"
            self.pdf_processor.save_processed_content(processed_pdfs, str(pdf_output_file))
            
            total_pdfs = len(processed_pdfs)
            total_chunks = sum(pdf.get('stats', {}).get('total_chunks', 0) for pdf in processed_pdfs)
            logger.info(f"✅ Processed {total_pdfs} PDFs with {total_chunks} training chunks")
        else:
            logger.warning("⚠️ No PDFs were successfully processed")
        
        return processed_pdfs
    
    def process_manual_data(self) -> Dict[str, Any]:
        """
        Step 3: Process manually curated data
        """
        logger.info("📝 Step 3: Processing manual data...")
        
        manual_data = {
            "faqs": [],
            "investment_data": [],
            "fund_performance": [],
            "company_info": []
        }
        
        # Load existing processed data if available
        processed_file = self.data_dir / "processed_data.json"
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Extract manual data sections
                manual_data["faqs"] = existing_data.get("qa_pairs", [])
                manual_data["company_info"] = existing_data.get("documents", [])
                
                logger.info(f"✅ Loaded existing manual data: {len(manual_data['faqs'])} Q&As")
            except Exception as e:
                logger.error(f"❌ Failed to load existing data: {e}")
        
        # Add sample financial data if none exists
        if not manual_data["fund_performance"]:
            manual_data["fund_performance"] = [
                {
                    "fund_name": "Arvocap Money Market Fund",
                    "performance": "16.5% average annual return",
                    "risk_level": "Low risk",
                    "description": "Conservative investment option for capital preservation"
                },
                {
                    "fund_name": "Thamani Equity Fund", 
                    "performance": "Aggressive growth strategy",
                    "risk_level": "High risk, high reward",
                    "description": "Equity-focused fund for long-term wealth building"
                }
            ]
        
        return manual_data
    
    def create_training_dataset(self, scraped_data: Dict, pdf_data: List[Dict], manual_data: Dict) -> Dict[str, Any]:
        """
        Step 4: Combine and process all data sources into training format
        """
        logger.info("🔄 Step 4: Creating unified training dataset...")
        
        all_documents = []
        all_qa_pairs = []
        all_chunks = []
        
        # Process web scraped data
        for url, content in scraped_data.items():
            if isinstance(content, dict) and 'text' in content:
                doc = {
                    "source": url,
                    "type": "web",
                    "content": content['text'],
                    "metadata": {"url": url, "scraped_at": content.get('timestamp')}
                }
                all_documents.append(doc)
                
                # Create chunks
                chunks = self.text_processor.chunk_text(content['text'])
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "content": chunk,
                        "source": url,
                        "type": "web",
                        "chunk_id": f"{url}_chunk_{i}"
                    })
        
        # Process PDF data
        for pdf in pdf_data:
            content = pdf.get('content', {})
            chunks = content.get('chunks', [])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "source": pdf.get('source', 'unknown'),
                    "type": "pdf",
                    "chunk_id": f"{pdf.get('metadata', {}).get('filename', 'unknown')}_chunk_{i}",
                    "metadata": pdf.get('metadata', {})
                })
        
        # Process manual data
        all_qa_pairs.extend(manual_data.get("faqs", []))
        
        for fund_info in manual_data.get("fund_performance", []):
            doc = {
                "source": "manual_fund_data",
                "type": "fund_info",
                "content": f"{fund_info['fund_name']}: {fund_info['description']}. Performance: {fund_info['performance']}. Risk Level: {fund_info['risk_level']}",
                "metadata": fund_info
            }
            all_documents.append(doc)
        
        # Create final training dataset
        training_dataset = {
            "documents": all_documents,
            "qa_pairs": all_qa_pairs, 
            "chunks": all_chunks,
            "stats": {
                "total_documents": len(all_documents),
                "total_qa_pairs": len(all_qa_pairs),
                "total_chunks": len(all_chunks),
                "sources": {
                    "web": len([d for d in all_documents if d['type'] == 'web']),
                    "pdf": len([c for c in all_chunks if c['type'] == 'pdf']),
                    "manual": len([d for d in all_documents if d['type'] == 'fund_info'])
                }
            },
            "created_at": self.text_processor.get_timestamp()
        }
        
        # Save training dataset
        output_file = self.data_dir / "processed_data.json" 
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Training dataset created with {len(all_chunks)} chunks")
        logger.info(f"   - Web sources: {training_dataset['stats']['sources']['web']}")
        logger.info(f"   - PDF chunks: {training_dataset['stats']['sources']['pdf']} ")
        logger.info(f"   - Manual entries: {training_dataset['stats']['sources']['manual']}")
        
        return training_dataset
    
    def build_vector_database(self, training_dataset: Dict[str, Any]) -> bool:
        """
        Step 5: Build/update vector database with all training data
        """
        logger.info("🔍 Step 5: Building vector database...")
        
        try:
            # Clear existing data
            self.knowledge_base.clear_database()
            
            # Prepare all documents for batch processing
            all_documents_for_db = []
            
            # Process chunks in batch
            chunks = training_dataset.get("chunks", [])
            for chunk in chunks:
                doc_data = {
                    'content': chunk["content"],
                    'metadata': {
                        "source": chunk["source"],
                        "type": chunk["type"],
                        "chunk_id": chunk["chunk_id"]
                    }
                }
                all_documents_for_db.append(doc_data)
            
            # Process Q&A pairs in batch
            qa_pairs = training_dataset.get("qa_pairs", [])
            for qa in qa_pairs:
                doc_data = {
                    'content': f"Q: {qa.get('question', '')} A: {qa.get('answer', '')}",
                    'metadata': {
                        "source": "manual_qa",
                        "type": "qa_pair",
                        "question": qa.get('question', ''),
                        "answer": qa.get('answer', '')
                    }
                }
                all_documents_for_db.append(doc_data)
            
            # Add all documents in a single batch operation
            if all_documents_for_db:
                logger.info(f"📥 Adding {len(all_documents_for_db)} documents to vector database in batch...")
                self.knowledge_base.vector_db.add_documents(all_documents_for_db)
            
            total_docs = len(chunks) + len(qa_pairs)
            logger.info(f"✅ Vector database built with {total_docs} documents")
            
            # Test the database
            collection_size = self.knowledge_base.get_collection_size()
            logger.info(f"   📊 Database collection size: {collection_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build vector database: {e}")
            return False
    
    def train_chatbot(self, training_dataset: Dict[str, Any]) -> bool:
        """
        Step 6: Train/update the chatbot model
        """
        logger.info("🤖 Step 6: Training chatbot...")
        
        try:
            # Use OpenAI-based training (no local model training needed)
            # The vector database serves as the knowledge base
            
            # Update trainer with new data
            processed_file = self.data_dir / "processed_data.json"
            success = self.trainer.train_on_data(str(processed_file))
            
            if success:
                logger.info("✅ Chatbot training completed successfully")
                return True
            else:
                logger.error("❌ Chatbot training failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Chatbot training error: {e}")
            return False
    
    def run_full_pipeline(self, include_web: bool = True, include_pdfs: bool = True, 
                         urls: List[str] = None, pdf_dir: str = None) -> bool:
        """
        Run the complete training pipeline
        """
        logger.info("🚀 Starting Arvocap Chatbot Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Collect web data
            scraped_data = {}
            if include_web:
                scraped_data = self.collect_web_data(urls)
            
            # Step 2: Process PDFs
            pdf_data = []
            if include_pdfs:
                pdf_data = self.collect_pdf_data(pdf_dir)
            
            # Step 3: Process manual data
            manual_data = self.process_manual_data()
            
            # Step 4: Create training dataset
            training_dataset = self.create_training_dataset(scraped_data, pdf_data, manual_data)
            
            # Step 5: Build vector database
            vector_success = self.build_vector_database(training_dataset)
            if not vector_success:
                logger.error("❌ Vector database creation failed")
                return False
            
            # Step 6: Train chatbot
            training_success = self.train_chatbot(training_dataset)
            if not training_success:
                logger.error("❌ Chatbot training failed")
                return False
            
            logger.info("=" * 60)
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"📊 Final Stats:")
            stats = training_dataset.get('stats', {})
            logger.info(f"   - Total documents: {stats.get('total_documents', 0)}")
            logger.info(f"   - Total Q&A pairs: {stats.get('total_qa_pairs', 0)}")
            logger.info(f"   - Total chunks: {stats.get('total_chunks', 0)}")
            logger.info(f"   - Vector database size: {self.knowledge_base.get_collection_size()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False

def main():
    """
    Command-line interface for the training pipeline
    """
    parser = argparse.ArgumentParser(description="Arvocap Chatbot Training Pipeline")
    
    parser.add_argument('--data-dir', default=DATA_PATH, 
                       help='Directory for training data')
    parser.add_argument('--pdf-dir', 
                       help='Directory containing PDF files')
    parser.add_argument('--no-web', action='store_true',
                       help='Skip web scraping')
    parser.add_argument('--no-pdfs', action='store_true', 
                       help='Skip PDF processing')
    parser.add_argument('--urls', nargs='+',
                       help='Specific URLs to scrape')
    parser.add_argument('--pdf-only', action='store_true',
                       help='Process only PDFs')
    parser.add_argument('--rebuild-db', action='store_true',
                       help='Force rebuild vector database')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ArvocapTrainingPipeline(args.data_dir)
    
    # Configure options
    include_web = not args.no_web and not args.pdf_only
    include_pdfs = not args.no_pdfs
    
    if args.pdf_only:
        include_web = False
        include_pdfs = True
    
    # Run pipeline
    success = pipeline.run_full_pipeline(
        include_web=include_web,
        include_pdfs=include_pdfs,
        urls=args.urls,
        pdf_dir=args.pdf_dir
    )
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now:")
        print("1. Test the CLI: python chat_cli.py")
        print("2. Start the API server: python api_server.py")
        print("3. Use the web interface")
    else:
        print("\n❌ Training failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
