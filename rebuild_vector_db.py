#!/usr/bin/env python3
"""
Complete Vector Database Rebuild Script  
Properly loads all 392 PDF chunks into FAISS vector database
"""

import json
import logging
import os
from pathlib import Path
from vector_database import VectorDatabase, ChatbotKnowledgeBase
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_vector_database():
    """
    Completely rebuild the FAISS vector database with all PDF chunks
    """
    try:
        logger.info("🔧 Starting complete FAISS vector database rebuild...")
        
        # Delete existing FAISS vector_db directories if they exist
        for db_path in ["vector_db", "vector_db_faiss"]:
            path = Path(db_path)
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"✅ Deleted existing {db_path}")
        
        # Create fresh FAISS vector database
        vector_db = VectorDatabase("vector_db_faiss")
        logger.info("✅ Created fresh FAISS vector database")
        
        # Load processed PDF data
        logger.info("📄 Loading processed PDF data...")
        with open('data/processed_pdfs.json', 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
        
        total_chunks = 0
        total_pdfs = len(pdf_data['pdfs'])
        
        logger.info(f"📊 Found {total_pdfs} PDFs to process")
        
        # Process each PDF and add ALL chunks individually
        for i, pdf in enumerate(pdf_data['pdfs']):
            filename = pdf['metadata']['filename']
            logger.info(f"📄 Processing PDF {i+1}/{total_pdfs}: {filename}")
            
            # Get all chunks from this PDF
            chunks = pdf['content']['chunks']
            logger.info(f"   Found {len(chunks)} chunks in {filename}")
            
            # Prepare documents for this PDF
            documents = []
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) > 20:  # Only add meaningful chunks
                    doc = {
                        'content': chunk,
                        'url': pdf['source'],
                        'title': filename,
                        'meta_description': f"Chunk {j+1} from {filename}",
                        'keywords': []
                    }
                    documents.append(doc)
            
            # Add documents to vector database
            if documents:
                vector_db.add_documents(documents)
                total_chunks += len(documents)
                logger.info(f"   ✅ Added {len(documents)} chunks from {filename}")
        
        logger.info(f"🎉 Successfully added {total_chunks} chunks from {total_pdfs} PDFs")
        
        # Verify the result
        kb = ChatbotKnowledgeBase()
        final_size = kb.get_collection_size()
        logger.info(f"🔍 Final vector database size: {final_size} documents")
        
        # Test search functionality
        logger.info("🔍 Testing search functionality...")
        results = kb.search_similar_content("CEO Monicah", max_results=3)
        logger.info(f"   Found {len(results)} results for CEO search")
        
        if results:
            logger.info(f"   Sample result: {results[0]['content'][:100]}...")
        
        return total_chunks, final_size
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding vector database: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

if __name__ == "__main__":
    print("🔧 Complete Vector Database Rebuild")
    print("=" * 50)
    
    chunks_added, final_size = rebuild_vector_database()
    
    if chunks_added > 0:
        print(f"\n🎉 SUCCESS!")
        print(f"   - Processed {chunks_added} PDF chunks")
        print(f"   - Final database size: {final_size} documents")
        print(f"   - Your chatbot now has access to ALL PDF content!")
    else:
        print(f"\n❌ FAILED!")
        print(f"   - Check the logs for errors")
