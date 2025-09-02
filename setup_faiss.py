#!/usr/bin/env python3
"""
FAISS Vector Database Setup and Test Script
Loads all 392 PDF chunks using FAISS for better performance
"""

import logging
from vector_database import ChatbotKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_faiss_database():
    """
    Setup FAISS vector database and load all PDF chunks
    """
    try:
        print("🚀 Setting up FAISS Vector Database")
        print("=" * 50)
        
        # Initialize knowledge base (will use FAISS if available)
        logger.info("Initializing knowledge base...")
        kb = ChatbotKnowledgeBase()
        
        # Check if using FAISS
        if hasattr(kb, 'using_faiss') and kb.using_faiss:
            print("✅ Using FAISS for vector database")
        else:
            print("⚠️ Falling back to ChromaDB")
        
        # Load all PDF chunks
        logger.info("Loading all PDF chunks...")
        total_chunks = kb.load_all_pdf_chunks()
        
        if total_chunks > 0:
            print(f"🎉 SUCCESS! Loaded {total_chunks} PDF chunks")
            
            # Verify the knowledge base size
            final_size = kb.get_collection_size()
            print(f"📊 Knowledge base now contains: {final_size} documents")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            test_queries = [
                "Who is the CEO?",
                "Money Market Fund performance",
                "Arvocap Asset Managers"
            ]
            
            for query in test_queries:
                results = kb.search_similar_content(query, max_results=2)
                print(f"   Query: '{query}' -> {len(results)} results")
                if results:
                    print(f"     Top result: {results[0]['content'][:100]}...")
            
            print(f"\n✅ FAISS vector database setup complete!")
            print(f"   Your chatbot now has access to {total_chunks} PDF chunks!")
            
            return True
        else:
            print("❌ Failed to load PDF chunks")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error setting up FAISS database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup_faiss_database()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Start the API server: python api_server.py")
        print("2. Test the frontend connection")
        print("3. Ask questions about your PDF content!")
    else:
        print("\n❌ Setup failed. Check the error messages above.")
