#!/usr/bin/env python3
"""
Quick test to verify our chatbot training setup is working.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import requests
        print("✅ requests")
        
        import bs4
        print("✅ beautifulsoup4")
        
        import openai
        print("✅ openai")
        
        import pandas
        print("✅ pandas")
        
        import numpy
        print("✅ numpy")
        
        import sklearn
        print("✅ scikit-learn")
        
        import torch
        print("✅ pytorch")
        
        import transformers
        print("✅ transformers")
        
        try:
            import faiss
            print("✅ faiss-cpu (primary vector database)")
        except Exception as e:
            print(f"⚠️  faiss-cpu: {e}")
        
        try:
            import chromadb
            print("✅ chromadb (fallback)")
        except Exception as e:
            print(f"⚠️  chromadb: {e}")
        
        print("\n🎉 All core packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_web_scraping():
    """Test basic web scraping functionality."""
    print("\n🌐 Testing web scraping...")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Test simple GET request
        response = requests.get("https://httpbin.org/html", timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            if soup.find('h1'):
                print("✅ Basic web scraping works!")
                return True
        
        print("⚠️  Web scraping test failed")
        return False
        
    except Exception as e:
        print(f"❌ Web scraping error: {e}")
        return False

def test_text_processing():
    """Test text processing functionality."""
    print("\n📝 Testing text processing...")
    
    try:
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Test text cleaning
        test_text = "Hello World! This is a test... 123"
        cleaned = re.sub(r'[^a-zA-Z\s]', '', test_text).lower()
        
        # Test TF-IDF
        texts = ["hello world", "world test", "test hello"]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        
        if vectors.shape[0] == 3:
            print("✅ Text processing works!")
            return True
        
        print("⚠️  Text processing test failed")
        return False
        
    except Exception as e:
        print(f"❌ Text processing error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Chatbot Training Environment Setup\n")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_web_scraping())
    results.append(test_text_processing())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 ALL TESTS PASSED! Your environment is ready for chatbot training!")
        print("\n📋 Next steps:")
        print("1. Set up your .env file with API keys")
        print("2. Run: python main.py --scrape-url https://example.com")
        print("3. Train your chatbot with the scraped data")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return all(results)

if __name__ == "__main__":
    main()
