import nltk
import spacy
import re
from typing import List, Dict, Tuple
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from config import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH, CHUNK_SIZE, CHUNK_OVERLAP

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

class TextProcessor:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def setup_spacy(self):
        """Setup spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove very long words (likely corrupted)
        words = text.split()
        words = [word for word in words if len(word) < 50]
        text = ' '.join(words)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                   overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                chunk_end = text.rfind('.', start, end)
                if chunk_end == -1:
                    chunk_end = text.rfind('!', start, end)
                if chunk_end == -1:
                    chunk_end = text.rfind('?', start, end)
                if chunk_end != -1 and chunk_end > start + chunk_size // 2:
                    end = chunk_end + 1
            
            chunk = text[start:end].strip()
            if len(chunk) >= MIN_TEXT_LENGTH:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            # Use spaCy for better keyword extraction if available
            if self.nlp:
                doc = self.nlp(text)
                keywords = []
                for token in doc:
                    if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                        not token.is_stop and 
                        not token.is_punct and 
                        len(token.text) > 2):
                        keywords.append(token.lemma_.lower())
                
                # Remove duplicates and return top keywords
                keywords = list(set(keywords))
                return keywords[:max_keywords]
            
            # Fallback to simple TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_keywords,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in keyword_scores if score > 0]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """Cluster texts based on similarity"""
        try:
            if len(texts) < n_clusters:
                n_clusters = len(texts)
            
            embeddings = self.generate_embeddings(texts)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Organize results
            clustered_texts = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_texts:
                    clustered_texts[cluster_id] = []
                clustered_texts[cluster_id].append({
                    'text': texts[i],
                    'index': i
                })
            
            return {
                'clusters': clustered_texts,
                'centroids': kmeans.cluster_centers_,
                'labels': clusters.tolist() if hasattr(clusters, 'tolist') else list(clusters)
            }
            
        except Exception as e:
            logger.error(f"Error clustering texts: {e}")
            return {}
    
    def extract_qa_pairs(self, text: str) -> List[Dict]:
        """Extract potential Q&A pairs from text"""
        qa_pairs = []
        
        # Look for FAQ patterns
        faq_patterns = [
            r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)',
            r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)',
            r'(\d+\.\s*.+?\?)\s*(.+?)(?=\d+\.|$)',
        ]
        
        for pattern in faq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) == 2:
                    question = self.clean_text(match[0])
                    answer = self.clean_text(match[1])
                    
                    if (len(question) > 10 and len(answer) > 20 and
                        question.endswith('?')):
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'keywords': self.extract_keywords(f"{question} {answer}")
                        })
        
        return qa_pairs
    
    def process_scraped_data(self, scraped_data: List[Dict]) -> Dict:
        """Process scraped data for training"""
        processed_data = {
            'documents': [],
            'qa_pairs': [],
            'keywords': [],
            'clusters': {}
        }
        
        all_texts = []
        
        for item in scraped_data:
            content = item.get('content', '')
            if len(content) < MIN_TEXT_LENGTH:
                continue
            
            # Clean content
            cleaned_content = self.clean_text(content)
            
            # Chunk content
            chunks = self.chunk_text(cleaned_content)
            
            for chunk in chunks:
                if len(chunk) >= MIN_TEXT_LENGTH:
                    keywords = self.extract_keywords(chunk)
                    
                    doc = {
                        'url': str(item.get('url', '')),
                        'title': str(item.get('title', '')),
                        'content': str(chunk),
                        'keywords': [str(kw) for kw in keywords],  # Ensure keywords are strings
                        'meta_description': str(item.get('meta_description', ''))
                    }
                    
                    processed_data['documents'].append(convert_to_native_types(doc))
                    all_texts.append(chunk)
                    processed_data['keywords'].extend(keywords)
            
            # Extract Q&A pairs
            qa_pairs = self.extract_qa_pairs(cleaned_content)
            processed_data['qa_pairs'].extend(qa_pairs)
        
        # Remove duplicate keywords and convert to native types
        processed_data['keywords'] = [str(kw) for kw in list(set(processed_data['keywords']))]
        
        # Cluster documents
        if all_texts:
            clusters = self.cluster_texts(all_texts)
            processed_data['clusters'] = convert_to_native_types(clusters)
        
        # Convert Q&A pairs to native types
        processed_data['qa_pairs'] = [convert_to_native_types(qa) for qa in processed_data['qa_pairs']]
        
        return convert_to_native_types(processed_data)

def main():
    """Example usage"""
    processor = TextProcessor()
    
    # Load scraped data
    try:
        with open('scraped_data.json', 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        
        print(f"Processing {len(scraped_data)} scraped pages...")
        processed_data = processor.process_scraped_data(scraped_data)
        
        # Save processed data
        with open('processed_data.json', 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            if 'clusters' in processed_data and 'centroids' in processed_data['clusters']:
                centroids = processed_data['clusters']['centroids']
                if hasattr(centroids, 'tolist'):
                    processed_data['clusters']['centroids'] = centroids.tolist()
                else:
                    processed_data['clusters']['centroids'] = list(centroids)
            
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed data saved:")
        print(f"- Documents: {len(processed_data['documents'])}")
        print(f"- Q&A pairs: {len(processed_data['qa_pairs'])}")
        print(f"- Unique keywords: {len(processed_data['keywords'])}")
        
    except FileNotFoundError:
        print("scraped_data.json not found. Run web_scraper.py first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
