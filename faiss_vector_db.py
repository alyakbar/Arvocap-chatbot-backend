#!/usr/bin/env python3
"""
FAISS-based Vector Database Implementation
Improved vector database using FAISS for better performance and proper PDF chunk loading
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorDatabase:
    """
    FAISS-based vector database for storing and searching document embeddings
    """
    
    def __init__(self, db_path: str = "vector_db_faiss", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # FAISS index
        self.index = None
        self.documents = []  # Store document content
        self.metadata = []   # Store document metadata
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_file = self.db_path / "faiss.index"
        docs_file = self.db_path / "documents.pkl"
        meta_file = self.db_path / "metadata.pkl"
        
        if index_file.exists() and docs_file.exists() and meta_file.exists():
            try:
                # Load existing index
                self.index = faiss.read_index(str(index_file))
                
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(meta_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded existing FAISS index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.documents = []
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def _save_index(self):
        """Save index and data to disk"""
        try:
            index_file = self.db_path / "faiss.index"
            docs_file = self.db_path / "documents.pkl"
            meta_file = self.db_path / "metadata.pkl"
            
            faiss.write_index(self.index, str(index_file))
            
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(meta_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info("Saved FAISS index to disk")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector database"""
        try:
            if not documents:
                return
            
            texts = []
            metadatas = []
            
            for doc in documents:
                content = doc.get('content', '')
                if not content:
                    continue
                
                texts.append(str(content))
                
                metadata = {
                    'source': str(doc.get('source', '')),
                    'title': str(doc.get('title', '')),
                    'url': str(doc.get('url', '')),
                    'type': str(doc.get('type', 'document')),
                    'chunk_id': doc.get('chunk_id', len(self.documents))
                }
                metadatas.append(metadata)
            
            if texts:
                # Generate embeddings
                embeddings = self.embedding_model.encode(texts)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.index.add(embeddings.astype('float32'))
                
                # Store documents and metadata
                self.documents.extend(texts)
                self.metadata.extend(metadatas)
                
                logger.info(f"Added {len(texts)} documents to FAISS index")
                
                # Save to disk
                self._save_index()
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def search(self, query: str, max_results: int = 5) -> Dict:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                return {'documents': [], 'metadatas': [], 'distances': []}
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding.astype('float32'), min(max_results, self.index.ntotal))
            
            # Format results
            documents = []
            metadatas = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    documents.append(self.documents[idx])
                    metadatas.append(self.metadata[idx])
                    result_distances.append(float(distances[0][i]))
            
            return {
                'documents': documents,
                'metadatas': metadatas,
                'distances': result_distances
            }
        
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_collection_size(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def clear_database(self):
        """Clear all data and create fresh index"""
        self._create_new_index()
        self._save_index()
        logger.info("Cleared FAISS database")

class FAISSChatbotKnowledgeBase:
    """
    FAISS-based knowledge base for the chatbot
    """
    
    def __init__(self, db_path: str = "vector_db_faiss"):
        self.vector_db = FAISSVectorDatabase(db_path)
    
    def add_document(self, content: str, metadata: Dict = None):
        """Add a single document to the knowledge base"""
        doc = {
            'content': content,
            'source': metadata.get('source', '') if metadata else '',
            'type': metadata.get('type', 'document') if metadata else 'document',
            'chunk_id': metadata.get('chunk_id', '') if metadata else ''
        }
        self.vector_db.add_documents([doc])
    
    def search_similar_content(self, query: str, max_results: int = 5):
        """Search for similar content and return formatted results"""
        try:
            results = self.vector_db.search(query, max_results)
            
            formatted = []
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            distances = results.get('distances', [])
            
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                # Convert distance to score (higher is better)
                distance = distances[i] if i < len(distances) else 0.0
                score = max(0.0, float(distance))  # FAISS returns similarity scores already
                
                formatted.append({
                    "content": doc,
                    "metadata": metadata,
                    "score": score
                })
            
            return formatted
        except Exception as e:
            logger.error(f"Error in search_similar_content: {e}")
            return []
    
    def find_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Find relevant context for a query"""
        results = self.search_similar_content(query, max_results=5)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            doc = result['content']
            if current_length + len(doc) > max_context_length:
                break
            
            context_parts.append(doc)
            current_length += len(doc)
        
        return "\n\n".join(context_parts)
    
    def get_collection_size(self) -> int:
        """Get total number of documents in knowledge base"""
        return self.vector_db.get_collection_size()
    
    def clear_database(self):
        """Clear the entire knowledge base"""
        self.vector_db.clear_database()
    
    def load_pdf_chunks(self, processed_pdfs_file: str = "data/processed_pdfs.json"):
        """Load all PDF chunks into the knowledge base"""
        try:
            logger.info(f"Loading PDF chunks from {processed_pdfs_file}")
            
            with open(processed_pdfs_file, 'r', encoding='utf-8') as f:
                pdf_data = json.load(f)
            
            total_chunks = 0
            total_pdfs = len(pdf_data['pdfs'])
            
            # Clear existing data first
            self.clear_database()
            
            # Process each PDF
            for i, pdf in enumerate(pdf_data['pdfs']):
                logger.info(f"Loading PDF {i+1}/{total_pdfs}: {pdf['metadata']['filename']}")
                
                # Add each chunk as a separate document
                chunks = pdf['content']['chunks']
                
                # Prepare documents for batch insert
                documents = []
                for j, chunk in enumerate(chunks):
                    doc = {
                        'content': chunk,
                        'source': pdf['source'],
                        'title': pdf['metadata']['filename'],
                        'type': 'pdf_chunk',
                        'chunk_id': f"{pdf['metadata']['filename']}_chunk_{j}"
                    }
                    documents.append(doc)
                
                # Add documents in batch
                if documents:
                    self.vector_db.add_documents(documents)
                    total_chunks += len(documents)
            
            logger.info(f"✅ Successfully loaded {total_chunks} chunks from {total_pdfs} PDFs")
            return total_chunks
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF chunks: {e}")
            return 0

def main():
    """Test the FAISS knowledge base"""
    print("🔧 Testing FAISS Knowledge Base")
    print("=" * 40)
    
    # Initialize knowledge base
    kb = FAISSChatbotKnowledgeBase()
    
    # Load PDF chunks
    total_chunks = kb.load_pdf_chunks()
    
    if total_chunks > 0:
        print(f"✅ Loaded {total_chunks} PDF chunks")
        print(f"📊 Knowledge base size: {kb.get_collection_size()}")
        
        # Test search
        query = "Who is the CEO?"
        print(f"\n🔍 Testing search: '{query}'")
        results = kb.search_similar_content(query, max_results=3)
        
        for i, result in enumerate(results):
            print(f"Result {i+1} (score: {result['score']:.3f}):")
            print(f"  Content: {result['content'][:100]}...")
            print(f"  Source: {result['metadata'].get('title', 'Unknown')}")
    else:
        print("❌ Failed to load PDF chunks")

if __name__ == "__main__":
    main()
