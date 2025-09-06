import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _to_str_list(x):
    return [str(i) for i in x] if isinstance(x, list) else []

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class VectorDatabase:
    """
    FAISS-based vector database implementation (single canonical version)
    - Persistent FAISS index, documents, metadata, and dedup hashes
    - Cosine similarity using normalized vectors (IndexFlatIP)
    - Flat return format for searches
    """
    def __init__(self, db_path: str = VECTOR_DB_PATH, collection_name: str = COLLECTION_NAME, embedding_model: str = EMBEDDING_MODEL):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.embedding_model = SentenceTransformer(embedding_model)
        try:
            self.embedding_dim = int(getattr(self.embedding_model, "get_sentence_embedding_dimension", lambda: None)() or 0)
        except Exception:
            self.embedding_dim = 0
        if not self.embedding_dim:
            # Fallback: compute from a sample
            self.embedding_dim = int(self.embedding_model.encode(["sample"]).shape[1])

        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []
        self.doc_hashes: Set[str] = set()

        self._index_file = self.db_path / "faiss.index"
        self._docs_file = self.db_path / "documents.pkl"
        self._meta_file = self.db_path / "metadata.pkl"
        self._hashes_file = self.db_path / "hashes.pkl"

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index and data or create a new one."""
        if self._index_file.exists() and self._docs_file.exists() and self._meta_file.exists():
            try:
                self.index = faiss.read_index(str(self._index_file))
                with open(self._docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                with open(self._meta_file, "rb") as f:
                    self.metadata = pickle.load(f)
                if self._hashes_file.exists():
                    with open(self._hashes_file, "rb") as f:
                        self.doc_hashes = pickle.load(f)
                else:
                    # Build hashes from existing docs if missing
                    self.doc_hashes = { _hash_text(t) for t in self.documents }
                logger.info(f"✅ Loaded FAISS index with {len(self.documents)} documents")
                return
            except Exception as e:
                logger.error(f"Failed to load FAISS index, creating new: {e}")

        self._create_new_index()

    def _create_new_index(self):
        """Create a new empty FAISS index."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # cosine similarity with normalized vectors
        self.documents = []
        self.metadata = []
        self.doc_hashes = set()
        logger.info("✅ Created new FAISS index")

    def _save_index(self):
        """Persist index, documents, metadata, and hashes."""
        try:
            faiss.write_index(self.index, str(self._index_file))
            with open(self._docs_file, "wb") as f:
                pickle.dump(self.documents, f)
            with open(self._meta_file, "wb") as f:
                pickle.dump(self.metadata, f)
            with open(self._hashes_file, "wb") as f:
                pickle.dump(self.doc_hashes, f)
            logger.info(f"💾 Saved FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector database (deduplicated by content hash).
        Input doc keys supported:
          - 'content' (required)
          - optional metadata: 'url','title','source','type','keywords','meta_description','chunk_id'
        """
        try:
            if not documents:
                return

            texts: List[str] = []
            metadatas: List[Dict] = []

            for doc in documents:
                content = str(doc.get("content", "")).strip()
                if not content or len(content) < 10:
                    continue
                h = _hash_text(content)
                if h in self.doc_hashes:
                    continue  # skip duplicates

                texts.append(content)
                md = {
                    "url": str(doc.get("url", "")),
                    "title": str(doc.get("title", "")),
                    "source": str(doc.get("source", "")),
                    "type": str(doc.get("type", "document")),
                    "keywords": _to_str_list(doc.get("keywords", [])),
                    "meta_description": str(doc.get("meta_description", "")),
                    "chunk_id": doc.get("chunk_id")
                }
                metadatas.append(md)
                self.doc_hashes.add(h)

            if not texts:
                return

            embeddings = self.embedding_model.encode(texts)
            faiss.normalize_L2(embeddings)  # cosine similarity with IP

            self.index.add(embeddings.astype("float32"))
            self.documents.extend(texts)
            self.metadata.extend(metadatas)

            logger.info(f"📄 Added {len(texts)} documents to FAISS (total: {len(self.documents)})")
            self._save_index()
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")

    def add_qa_pairs(self, qa_pairs: List[Dict]) -> None:
        """Add Q&A pairs; stored as combined 'Q: ... A: ...' content."""
        try:
            if not qa_pairs:
                return
            docs = []
            for qa in qa_pairs:
                q = str(qa.get("question", "")).strip()
                a = str(qa.get("answer", "")).strip()
                if not q or not a:
                    continue
                docs.append({
                    "content": f"Q: {q} A: {a}",
                    "type": "qa_pair",
                    "title": f"Q&A: {q}",
                    "source": "manual_qa",
                    "keywords": _to_str_list(qa.get("keywords", []))
                })
            self.add_documents(docs)
            logger.info(f"❓ Added {len(docs)} Q&A pairs")
        except Exception as e:
            logger.error(f"Error adding Q&A pairs to FAISS: {e}")

    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Search similar documents. Returns flat lists:
          {
            'documents': [str, ...],
            'metadatas': [dict, ...],
            'distances': [float, ...]  # cosine similarity, higher is better
          }
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return {"documents": [], "metadatas": [], "distances": []}

            query_emb = self.embedding_model.encode([query])
            faiss.normalize_L2(query_emb)
            k = min(int(n_results), self.index.ntotal)
            distances, indices = self.index.search(query_emb.astype("float32"), k)

            docs, metas, dists = [], [], []
            for i, idx in enumerate(indices[0]):
                if 0 <= int(idx) < len(self.documents):
                    docs.append(self.documents[idx])
                    metas.append(self.metadata[idx])
                    dists.append(float(distances[0][i]))
            return {"documents": docs, "metadatas": metas, "distances": dists}
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def get_collection_stats(self) -> Dict:
        """Get collection stats."""
        return {
            "total_documents": int(len(self.documents)),
            "index_size": int(self.index.ntotal if self.index is not None else 0),
            "embedding_dim": int(self.embedding_dim),
            "db_path": str(self.db_path)
        }

    def delete_collection(self):
        """Clear all data (use with care)."""
        try:
            self._create_new_index()
            self._save_index()
            logger.info("🗑️ Cleared FAISS collection")
        except Exception as e:
            logger.error(f"Error deleting FAISS collection: {e}")

class ChatbotKnowledgeBase:
    """
    Knowledge base on top of FAISS VectorDatabase.
    Provides higher-level helpers and consistent API to the rest of the app.
    """
    def __init__(self):
        logger.info("🚀 Initializing FAISS-based knowledge base")
        self.vector_db = VectorDatabase(VECTOR_DB_PATH)
        self.using_faiss = True
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def clear_database(self):
        """Clear the vector database."""
        try:
            self.vector_db.delete_collection()
            self.vector_db = VectorDatabase(VECTOR_DB_PATH)
            logger.info("Vector database cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")

    def add_document(self, content: str, metadata: Dict = None):
        """Add a single document with flattened metadata."""
        try:
            md = metadata or {}
            doc = {
                "content": content,
                "url": str(md.get("url", "")),
                "title": str(md.get("title", "")),
                "source": str(md.get("source", "")),
                "type": str(md.get("type", "document")),
                "keywords": _to_str_list(md.get("keywords", [])),
                "meta_description": str(md.get("meta_description", "")),
                "chunk_id": md.get("chunk_id")
            }
            self.vector_db.add_documents([doc])
        except Exception as e:
            logger.error(f"Error adding document: {e}")

    def get_collection_size(self) -> int:
        """Total number of documents/items in the collection."""
        try:
            stats = self.vector_db.get_collection_stats()
            return int(stats.get("total_documents", 0))
        except Exception:
            return 0

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge base."""
        try:
            stats = self.vector_db.get_collection_stats()
            
            # Count different types of content
            document_count = 0
            website_count = 0
            manual_count = 0
            
            for meta in self.vector_db.metadata:
                item_type = meta.get("type", "document")
                if item_type in ["pdf", "document", "pdf_chunk"]:
                    document_count += 1
                elif item_type in ["website", "web_page"]:
                    website_count += 1
                elif item_type in ["manual", "manual_entry"]:
                    manual_count += 1
            
            # Get last trained timestamp
            last_trained = None
            stats_file = self.vector_db.db_path / "last_trained.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        data = json.load(f)
                        last_trained = data.get("timestamp")
                except:
                    pass
            
            return {
                "document_count": document_count,
                "website_count": website_count,
                "manual_count": manual_count,
                "total_items": stats.get("total_documents", 0),
                "last_trained": last_trained
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "document_count": 0,
                "website_count": 0,
                "manual_count": 0,
                "total_items": 0,
                "last_trained": None
            }

    def get_all_items(self) -> List[Dict]:
        """Get all knowledge base items with metadata for admin interface."""
        try:
            items = []
            for i, (doc, meta) in enumerate(zip(self.vector_db.documents, self.vector_db.metadata)):
                items.append({
                    "id": str(i),  # Use index as ID
                    "type": meta.get("type", "document"),
                    "title": meta.get("title", "Untitled"),
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,  # Truncate for list view
                    "timestamp": meta.get("timestamp", "2024-01-01T00:00:00Z"),
                    "size": len(doc),
                    "status": "active",
                    "metadata": meta
                })
            return items
        except Exception as e:
            logger.error(f"Error getting all items: {e}")
            return []

    def delete_item(self, item_id: str) -> bool:
        """Delete a specific item by ID."""
        try:
            idx = int(item_id)
            if 0 <= idx < len(self.vector_db.documents):
                # Remove from all collections
                del self.vector_db.documents[idx]
                del self.vector_db.metadata[idx]
                
                # Rebuild FAISS index
                if self.vector_db.documents:
                    embeddings = self.vector_db.embedding_model.encode(self.vector_db.documents)
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    self.vector_db.index = faiss.IndexFlatIP(embeddings.shape[1])
                    self.vector_db.index.add(embeddings.astype('float32'))
                else:
                    self.vector_db.index = faiss.IndexFlatIP(self.vector_db.embedding_dim)
                
                # Save changes
                self.vector_db._save()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            return False

    def update_item(self, item_id: str, title: str, content: str) -> bool:
        """Update a specific item by ID."""
        try:
            idx = int(item_id)
            if 0 <= idx < len(self.vector_db.documents):
                # Update document and metadata
                self.vector_db.documents[idx] = content
                self.vector_db.metadata[idx]["title"] = title
                self.vector_db.metadata[idx]["timestamp"] = datetime.now().isoformat() + "Z"
                
                # Rebuild FAISS index
                embeddings = self.vector_db.embedding_model.encode(self.vector_db.documents)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                self.vector_db.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.vector_db.index.add(embeddings.astype('float32'))
                
                # Save changes
                self.vector_db._save()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating item {item_id}: {e}")
            return False

    def add_manual_entry(self, title: str, content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """Add a manual entry to the knowledge base with chunking."""
        try:
            from datetime import datetime
            from text_processor import TextProcessor
            import traceback
            # Process content into chunks
            processor = TextProcessor()
            chunks = processor.create_chunks(content, chunk_size, chunk_overlap)
            # Add each chunk as a document
            chunks_created = 0
            timestamp = datetime.now().isoformat() + "Z"
            for i, chunk in enumerate(chunks):
                doc = {
                    "content": chunk,
                    "title": f"{title} (Part {i+1})" if len(chunks) > 1 else title,
                    "type": "manual_entry",
                    "source": "manual",
                    "timestamp": timestamp,
                    "chunk_id": f"manual_{title.replace(' ', '_')}_{i}"
                }
                self.add_document(doc)
                chunks_created += 1
            self._update_last_trained()
            return chunks_created
        except Exception as e:
            logger.error(f"Error adding manual entry: {e}")
            logger.error(traceback.format_exc())
            return 0

    def _update_last_trained(self):
        """Update the last trained timestamp and statistics."""
        try:
            from datetime import datetime
            
            # Get current statistics
            stats = self.get_statistics()
            
            # Update with timestamp
            stats_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "total_documents": stats.get("total_items", 0),
                "web_pages": stats.get("document_count", 0),  # Most items are from web scraping
                "pdf_pages": 0,  # Would need to track this separately
                "website_count": stats.get("website_count", 0),
                "manual_count": stats.get("manual_count", 0)
            }
            
            stats_file = self.vector_db.db_path / "last_trained.json"
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
            logger.info(f"✅ Updated training stats: {stats_data['total_documents']} total documents")
            
        except Exception as e:
            logger.error(f"Error updating last trained timestamp: {e}")

    def search_similar_content(self, query: str, max_results: int = 5):
        """Return list of {content, metadata, score} with cosine similarity score."""
        try:
            res = self.vector_db.search(query, n_results=max_results)
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            dists = res.get("distances", [])

            formatted = []
            for i, doc in enumerate(docs):
                md = metas[i] if i < len(metas) else {}
                score = float(dists[i]) if i < len(dists) else 0.0  # cosine sim (higher is better)
                formatted.append({"content": doc, "metadata": md, "score": score})
            return formatted
        except Exception as e:
            logger.error(f"Error in search_similar_content: {e}")
            return []

    def load_training_data(self, processed_data_file: str):
        """Load processed training data into the knowledge base."""
        try:
            candidate = processed_data_file
            if not os.path.isabs(candidate):
                if not os.path.exists(candidate):
                    base_dir = os.path.dirname(__file__)
                    alt = os.path.join(base_dir, "data", processed_data_file)
                    if os.path.exists(alt):
                        candidate = alt
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("documents"):
                self.vector_db.add_documents(data["documents"])
            if data.get("qa_pairs"):
                self.vector_db.add_qa_pairs(data["qa_pairs"])

            logger.info("Training data loaded into knowledge base")
        except Exception as e:
            logger.error(f"Error loading training data: {e}")

    def retrain(self, processed_data_file: str = "processed_data.json", clear_existing: bool = False) -> dict:
        """
        Load data from a processed data file into the vector DB.
        By default, preserves existing knowledge and only adds new (deduplicated) items.
        Set clear_existing=True to fully rebuild.
        """
        try:
            if clear_existing:
                try:
                    self.vector_db.delete_collection()
                except Exception:
                    pass
                self.vector_db = VectorDatabase(VECTOR_DB_PATH)

            candidate = processed_data_file
            if not os.path.isabs(candidate):
                if not os.path.exists(candidate):
                    base_dir = os.path.dirname(__file__)
                    alt = os.path.join(base_dir, "data", processed_data_file)
                    if os.path.exists(alt):
                        candidate = alt

            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)

            added_docs = 0
            added_qas = 0
            if data.get("documents"):
                before = self.get_collection_size()
                self.vector_db.add_documents(data["documents"])
                after = self.get_collection_size()
                added_docs = max(0, after - before)
            if data.get("qa_pairs"):
                before = self.get_collection_size()
                self.vector_db.add_qa_pairs(data["qa_pairs"])
                after = self.get_collection_size()
                added_qas = max(0, after - before)

            stats = self.get_stats() or {}
            report = {
                "success": True,
                "added_documents": added_docs,
                "added_qa_pairs": added_qas,
                "total_documents": int(stats.get("total_documents", 0)),
                "cleared": bool(clear_existing)
            }
            
            # Update last trained timestamp
            self._update_last_trained()
            
            logger.info(f"Retrain completed: {report}")
            return report
        except FileNotFoundError:
            msg = "Processed data file not found. Run text_processor.py first."
            logger.error(msg)
            return {"success": False, "error": msg}
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            return {"success": False, "error": str(e)}

    def find_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Find relevant context by stitching top results until limit."""
        results = self.search_similar_content(query, max_results=5)
        parts, cur = [], 0
        for r in results:
            txt = r.get("content", "") or ""
            if not txt:
                continue
            if cur + len(txt) > max_context_length:
                break
            parts.append(txt)
            cur += len(txt)
        return "\n\n".join(parts)

    def load_all_pdf_chunks(self, processed_pdfs_file: str = "data/processed_pdfs.json"):
        """Load all PDF chunks efficiently into FAISS."""
        try:
            logger.info(f"🔄 Loading all PDF chunks from {processed_pdfs_file}")
            with open(processed_pdfs_file, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)

            total_chunks = 0
            # Support both list and dict formats
            pdfs_list = []
            if isinstance(pdf_data, list):
                pdfs_list = pdf_data
            elif isinstance(pdf_data, dict):
                pdfs_list = pdf_data.get("pdfs", []) or pdf_data.get("processed_pdfs", []) or []
            else:
                logger.warning("Unrecognized PDF data format; expected list or dict")

            total_pdfs = len(pdfs_list)
            logger.info(f"📊 Found {total_pdfs} PDFs to process")

            batch: List[Dict] = []
            for i, pdf in enumerate(pdfs_list):
                filename = (pdf.get("metadata", {}) or {}).get("filename", f"pdf_{i}")
                src = pdf.get("source", "")
                chunks = (pdf.get("content", {}) or {}).get("chunks", []) or []
                for j, chunk in enumerate(chunks):
                    if isinstance(chunk, str) and len(chunk.strip()) > 20:
                        batch.append({
                            "content": chunk,
                            "source": src,
                            "title": filename,
                            "url": src,
                            "type": "pdf_chunk",
                            "chunk_id": f"{filename}_chunk_{j}"
                        })
                        total_chunks += 1
                    if len(batch) >= 50:
                        self.vector_db.add_documents(batch)
                        batch = []
                logger.info(f"📄 Processed PDF {i+1}/{total_pdfs}: {filename}")

            if batch:
                self.vector_db.add_documents(batch)

            final_size = self.get_collection_size()
            logger.info(f"🎉 Loaded {total_chunks} chunks. Final KB size: {final_size}")
            return total_chunks
        except Exception as e:
            logger.error(f"❌ Error loading PDF chunks: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get knowledge base stats (pass-through from vector DB)."""
        return self.vector_db.get_collection_stats()

def main():
    """Manual test"""
    kb = ChatbotKnowledgeBase()
    try:
        kb.load_training_data("processed_data.json")
        query = input("Enter a test query: ")
        ctx = kb.find_relevant_context(query)
        print(f"\nRelevant context for '{query}':\n" + "-"*50)
        print(ctx)
        print("\nStats:", kb.get_stats())
    except FileNotFoundError:
        print("processed_data.json not found. Run text_processor.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
