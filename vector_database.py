import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from pathlib import Path

# FAISS imports
import faiss
import pickle

# ChromaDB imports (fallback only)
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import NotFoundError
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available, using FAISS only")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_native_types(data):
    """Convert numpy/pandas types to native Python types"""
    if data is None:
        return None
    elif isinstance(data, dict):
        return {str(k): convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, (np.int32, np.int64, np.integer)):
        return int(data)
    elif isinstance(data, (np.float32, np.float64, np.floating)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif hasattr(data, 'tolist'):  # Handle other array-like objects
        return data.tolist()
    else:
        return data

# Use FAISS as the primary VectorDatabase class
class VectorDatabase:
    """
    FAISS-based vector database implementation (now the default)
    """
    
    def __init__(self, db_path: str = "vector_db_faiss", collection_name: str = COLLECTION_NAME):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.collection_name = collection_name
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
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
                
                logger.info(f"✅ Loaded existing FAISS index with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.documents = []
        self.metadata = []
        logger.info("✅ Created new FAISS index")
    
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
            
            logger.info(f"💾 Saved FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the FAISS vector database"""
        try:
            if not documents:
                return
            
            texts = []
            metadatas = []
            
            for doc in documents:
                content = doc.get('content', '')
                if not content or len(content.strip()) < 10:
                    continue
                
                texts.append(str(content))
                
                # Convert all metadata to native Python types
                metadata = {
                    'url': str(doc.get('url', '')),
                    'title': str(doc.get('title', '')),
                    'source': str(doc.get('source', '')),
                    'type': str(doc.get('type', 'document')),
                    'keywords': doc.get('keywords', []),
                    'meta_description': str(doc.get('meta_description', ''))
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
                
                logger.info(f"📄 Added {len(texts)} documents to FAISS (total: {len(self.documents)})")
                
                # Save to disk
                self._save_index()
        
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
    
    def add_qa_pairs(self, qa_pairs: List[Dict]) -> None:
        """Add Q&A pairs to the FAISS vector database"""
        try:
            documents = []
            
            for qa in qa_pairs:
                question = str(qa.get('question', ''))
                answer = str(qa.get('answer', ''))
                
                if not question or not answer:
                    continue
                
                # Store both question and answer as searchable content
                combined_text = f"Q: {question} A: {answer}"
                
                doc = {
                    'content': combined_text,
                    'type': 'qa_pair',
                    'title': f"Q&A: {question}",
                    'source': 'manual_qa',
                    'keywords': qa.get('keywords', [])
                }
                documents.append(doc)
            
            if documents:
                self.add_documents(documents)
                logger.info(f"❓ Added {len(documents)} Q&A pairs to FAISS")
            
        except Exception as e:
            logger.error(f"Error adding Q&A pairs to FAISS: {e}")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents using FAISS"""
        try:
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            search_k = min(n_results, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Format results in ChromaDB-compatible format
            documents = []
            metadatas = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    documents.append(self.documents[idx])
                    metadatas.append(self.metadata[idx])
                    result_distances.append(float(distances[0][i]))
            
            return {
                'documents': [documents],  # ChromaDB format
                'metadatas': [metadatas],  # ChromaDB format  
                'distances': [result_distances]  # ChromaDB format
            }
        
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim
        }
    
    def delete_collection(self):
        """Delete/clear the collection"""
        try:
            self._create_new_index()
            self._save_index()
            logger.info("🗑️ Cleared FAISS collection")
        except Exception as e:
            logger.error(f"Error deleting FAISS collection: {e}")

# Keep the specialized FAISS class for advanced use
    """
    FAISS-based vector database implementation for better performance
    """
    
    def __init__(self, db_path: str = "vector_db_faiss", embedding_model: str = EMBEDDING_MODEL):
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
                if not content or len(content.strip()) < 10:
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
                
                logger.info(f"Added {len(texts)} documents to FAISS index (total: {len(self.documents)})")
                
                # Save to disk
                self._save_index()
        
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents"""
        try:
            if self.index.ntotal == 0:
                return {'documents': [], 'metadatas': [], 'distances': []}
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding.astype('float32'), min(n_results, self.index.ntotal))
            
            # Format results
            documents = []
            metadatas = []
            result_distances = []
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    documents.append([self.documents[idx]])  # ChromaDB format
                    metadatas.append([self.metadata[idx]])   # ChromaDB format
                    result_distances.append([float(distances[0][i])])  # ChromaDB format
            
            return {
                'documents': documents,
                'metadatas': metadatas,
                'distances': result_distances
            }
        
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0
        }
    
    def delete_collection(self):
        """Delete/clear the collection"""
        self._create_new_index()
        self._save_index()

class VectorDatabase:
    def __init__(self, db_path: str = VECTOR_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = None
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup or get the collection"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except (NotFoundError, ValueError):
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Chatbot knowledge base"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector database"""
        try:
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                if not content:
                    continue
                
                texts.append(str(content))
                
                # Convert all metadata to native Python types
                metadata = {
                    'url': str(doc.get('url', '')),
                    'title': str(doc.get('title', '')),
                    'keywords': json.dumps(doc.get('keywords', [])),
                    'meta_description': str(doc.get('meta_description', ''))
                }
                metadatas.append(convert_to_native_types(metadata))
                ids.append(f"doc_{int(i)}")
            
            if texts:
                # Generate embeddings
                embeddings = self.embedding_model.encode(texts).tolist()
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(texts)} documents to vector database")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def add_qa_pairs(self, qa_pairs: List[Dict]) -> None:
        """Add Q&A pairs to the vector database"""
        try:
            texts = []
            metadatas = []
            ids = []
            
            for i, qa in enumerate(qa_pairs):
                question = str(qa.get('question', ''))
                answer = str(qa.get('answer', ''))
                
                if not question or not answer:
                    continue
                
                # Store both question and answer as searchable content
                combined_text = f"Q: {question} A: {answer}"
                texts.append(combined_text)
                
                metadata = {
                    'type': 'qa_pair',
                    'question': question,
                    'answer': answer,
                    'keywords': json.dumps(qa.get('keywords', []))
                }
                metadatas.append(convert_to_native_types(metadata))
                ids.append(f"qa_{int(i)}")
            
            if texts:
                # Generate embeddings
                embeddings = self.embedding_model.encode(texts).tolist()
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(texts)} Q&A pairs to vector database")
            
        except Exception as e:
            logger.error(f"Error adding Q&A pairs: {e}")
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0]
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

class ChatbotKnowledgeBase:
    def __init__(self):
        """Initialize knowledge base with FAISS as primary database"""
        logger.info("🚀 Initializing FAISS-based knowledge base")
        self.vector_db = VectorDatabase("vector_db_faiss")
        self.using_faiss = True
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def clear_database(self):
        """Clear all data from the vector database"""
        try:
            self.vector_db.delete_collection()
            # Recreate the collection
            self.vector_db = VectorDatabase()
            logger.info("Vector database cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
    
    def add_document(self, content: str, metadata: Dict = None):
        """Add a single document to the knowledge base"""
        try:
            doc_data = {
                'content': content,
                'metadata': metadata or {}
            }
            self.vector_db.add_documents([doc_data])
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def get_collection_size(self) -> int:
        """Return total number of documents/items in the collection.
        This is used by the API server health/status endpoints."""
        try:
            stats = self.vector_db.get_collection_stats()
            return int(stats.get('total_documents', 0))
        except Exception:
            return 0

    def search_similar_content(self, query: str, max_results: int = 5):
        """Search vector DB and return a list of {content, metadata, score} dicts.
        Score is derived from distance when available (higher is better)."""
        try:
            results = self.vector_db.search(query, n_results=max_results)
            documents = results.get('documents', []) or []
            metadatas = results.get('metadatas', []) or []
            distances = results.get('distances', []) or []

            formatted = []
            for i, doc in enumerate(documents):
                md = metadatas[i] if i < len(metadatas) else {}
                # Convert distance (lower is better) into a score (higher is better) when present
                if i < len(distances) and distances[i] is not None:
                    try:
                        dist_val = float(distances[i])
                        score = 1.0 - dist_val
                    except Exception:
                        score = 0.0
                else:
                    score = 0.0

                formatted.append({
                    "content": doc,
                    "metadata": md,
                    "score": float(score)
                })

            return formatted
        except Exception as e:
            logger.error(f"Error in search_similar_content: {e}")
            return []

    def load_training_data(self, processed_data_file: str):
        """Load processed training data into the knowledge base"""
        try:
            # Resolve file path (try given path, then ./data/... relative to this file)
            candidate = processed_data_file
            if not os.path.isabs(candidate):
                if not os.path.exists(candidate):
                    base_dir = os.path.dirname(__file__)
                    alt = os.path.join(base_dir, 'data', processed_data_file)
                    if os.path.exists(alt):
                        candidate = alt
            with open(candidate, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add documents
            if 'documents' in data:
                self.vector_db.add_documents(data['documents'])
            
            # Add Q&A pairs
            if 'qa_pairs' in data:
                self.vector_db.add_qa_pairs(data['qa_pairs'])
            
            logger.info("Training data loaded into knowledge base")
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")

    def retrain(self, processed_data_file: str = 'processed_data.json') -> dict:
        """Rebuild the vector database from a processed data file.
        Returns a brief report with counts."""
        try:
            # Best-effort: delete collection and recreate to avoid duplicates
            try:
                self.vector_db.delete_collection()
            except Exception:
                pass
            # Recreate a fresh collection
            self.vector_db = VectorDatabase()
            # Resolve file path
            candidate = processed_data_file
            if not os.path.isabs(candidate):
                if not os.path.exists(candidate):
                    base_dir = os.path.dirname(__file__)
                    alt = os.path.join(base_dir, 'data', processed_data_file)
                    if os.path.exists(alt):
                        candidate = alt
            # Load training data
            with open(candidate, 'r', encoding='utf-8') as f:
                data = json.load(f)

            added_docs = 0
            added_qas = 0
            if 'documents' in data and data['documents']:
                self.vector_db.add_documents(data['documents'])
                added_docs = len(data['documents'])
            if 'qa_pairs' in data and data['qa_pairs']:
                self.vector_db.add_qa_pairs(data['qa_pairs'])
                added_qas = len(data['qa_pairs'])

            stats = self.get_stats() or {}
            report = {
                'success': True,
                'added_documents': added_docs,
                'added_qa_pairs': added_qas,
                'total_documents': int(stats.get('total_documents', 0))
            }
            logger.info(f"Retrain completed: {report}")
            return report
        except FileNotFoundError:
            msg = "Processed data file not found. Run text_processor.py first."
            logger.error(msg)
            return {'success': False, 'error': msg}
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def find_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Find relevant context for a query"""
        results = self.vector_db.search(query, n_results=3)
        
        context_parts = []
        current_length = 0
        
        for doc, metadata in zip(results['documents'], results['metadatas']):
            if current_length + len(doc) > max_context_length:
                break
            
            context_parts.append(doc)
            current_length += len(doc)
        
        return "\n\n".join(context_parts)
    
    def load_all_pdf_chunks(self, processed_pdfs_file: str = "data/processed_pdfs.json"):
        """Load all PDF chunks into the knowledge base using FAISS for better performance"""
        try:
            logger.info(f"🔄 Loading all PDF chunks from {processed_pdfs_file}")
            
            with open(processed_pdfs_file, 'r', encoding='utf-8') as f:
                pdf_data = json.load(f)
            
            total_chunks = 0
            total_pdfs = len(pdf_data['pdfs'])
            
            logger.info(f"📊 Found {total_pdfs} PDFs to process")
            
            # Clear existing data first
            self.clear_database()
            
            # Process each PDF
            batch_documents = []
            for i, pdf in enumerate(pdf_data['pdfs']):
                filename = pdf['metadata']['filename']
                logger.info(f"📄 Processing PDF {i+1}/{total_pdfs}: {filename}")
                
                # Get all chunks from this PDF
                chunks = pdf['content']['chunks']
                
                # Prepare documents for this PDF
                for j, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 20:  # Only add meaningful chunks
                        doc = {
                            'content': chunk,
                            'source': pdf['source'],
                            'title': filename,
                            'url': pdf['source'],
                            'type': 'pdf_chunk',
                            'chunk_id': f"{filename}_chunk_{j}"
                        }
                        batch_documents.append(doc)
                        total_chunks += 1
                
                # Add documents in batches of 50 for better performance
                if len(batch_documents) >= 50:
                    self.vector_db.add_documents(batch_documents)
                    logger.info(f"   ✅ Added batch of {len(batch_documents)} chunks")
                    batch_documents = []
            
            # Add remaining documents
            if batch_documents:
                self.vector_db.add_documents(batch_documents)
                logger.info(f"   ✅ Added final batch of {len(batch_documents)} chunks")
            
            logger.info(f"🎉 Successfully loaded {total_chunks} chunks from {total_pdfs} PDFs")
            
            # Verify the result
            final_size = self.get_collection_size()
            logger.info(f"🔍 Final knowledge base size: {final_size} documents")
            
            return total_chunks
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF chunks: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return self.vector_db.get_collection_stats()

def main():
    """Example usage"""
    kb = ChatbotKnowledgeBase()
    
    # Load processed data
    try:
        kb.load_training_data('processed_data.json')
        
        # Test search
        query = input("Enter a test query: ")
        context = kb.find_relevant_context(query)
        
        print(f"\nRelevant context for '{query}':")
        print("-" * 50)
        print(context)
        
        # Show stats
        stats = kb.get_stats()
        print(f"\nKnowledge base stats: {stats}")
        
    except FileNotFoundError:
        print("processed_data.json not found. Run text_processor.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
