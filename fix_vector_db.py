#!/usr/bin/env python3
"""
Fix Vector Database Script
Properly loads all PDF chunks into the vector database
"""

import json
import logging
from vector_database import ChatbotKnowledgeBase, VectorDatabase
from config import VECTOR_DB_PATH, COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_vector_database():
    """
    Load all PDF chunks properly into the vector database
    """
    try:
        # Initialize knowledge base
        logger.info("Initializing knowledge base...")
        kb = ChatbotKnowledgeBase()
        
        # Clear existing data
        logger.info("Clearing existing vector database...")
        try:
            kb.vector_db.delete_collection()
        except:
            pass  # Collection might not exist
        
        # Recreate fresh FAISS vector database
        kb.vector_db = VectorDatabase(VECTOR_DB_PATH)
        
        # Load processed PDF data
        logger.info("Loading processed PDF data...")
        with open('data/processed_pdfs.json', 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)

        # Support both list and dict formats
        if isinstance(pdf_data, list):
            pdfs = pdf_data
        else:
            pdfs = pdf_data.get('pdfs', []) or pdf_data.get('processed_pdfs', []) or []

        total_chunks = 0
        total_pdfs = len(pdfs)
        
        # Process each PDF and add all chunks to vector database
        batch = []
        for i, pdf in enumerate(pdfs):
            logger.info(f"Processing PDF {i+1}/{total_pdfs}: {pdf.get('metadata', {}).get('filename', 'unknown')}")
            
            chunks = (pdf.get('content', {}) or {}).get('chunks', []) or []
            for j, chunk in enumerate(chunks):
                doc_data = {
                    'content': chunk,
                    'url': pdf.get('source', ''),
                    'title': (pdf.get('metadata', {}) or {}).get('filename', ''),
                    'meta_description': f"PDF chunk {j+1}",
                    'keywords': [],
                    'type': 'pdf_chunk',
                    'source': pdf.get('source', ''),
                    'chunk_id': f"{(pdf.get('metadata', {}) or {}).get('filename', 'file')}_chunk_{j}"
                }
                batch.append(doc_data)
                total_chunks += 1

                if len(batch) >= 50:
                    kb.vector_db.add_documents(batch)
                    batch = []
        if batch:
            kb.vector_db.add_documents(batch)
        
        logger.info(f"✅ Successfully added {total_chunks} chunks from {total_pdfs} PDFs")
        
        # Verify the result
        final_size = kb.get_collection_size()
        logger.info(f"🔍 Final vector database size: {final_size} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error fixing vector database: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Vector Database with All PDF Chunks")
    print("=" * 50)
    
    success = fix_vector_database()
    
    if success:
        print("\n🎉 Vector database fixed successfully!")
        print("Your chatbot now has access to all PDF chunks!")
    else:
        print("\n❌ Failed to fix vector database. Check the logs for errors.")
