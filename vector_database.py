import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

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
        self.vector_db = VectorDatabase()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
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
