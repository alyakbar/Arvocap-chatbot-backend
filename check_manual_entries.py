#!/usr/bin/env python3
"""
Check manual entries in knowledge base
"""

from vector_database import ChatbotKnowledgeBase
from faiss_vector_db import FAISSVectorDatabase

def check_manual_entries():
    print("🔍 MANUAL ENTRY DIAGNOSIS")
    print("=" * 35)

    try:
        # Check knowledge base
        kb = ChatbotKnowledgeBase()
        items = kb.get_all_items()
        print(f"Total items in knowledge base: {len(items)}")

        # Filter manual entries
        manual_entries = [item for item in items if item.get('source_type') == 'manual']
        print(f"Manual entries: {len(manual_entries)}")

        if manual_entries:
            print("\n📝 MANUAL ENTRIES:")
            for i, entry in enumerate(manual_entries):
                title = entry.get('title', 'No title')
                content = entry.get('content', '')[:100]
                source = entry.get('source', 'Unknown')
                print(f"{i+1}. Title: {title}")
                print(f"   Content: {content}...")
                print(f"   Source: {source}")
                print()

        # Test if manual entries are in vector database
        print("🔍 CHECKING VECTOR DATABASE:")
        try:
            vector_db = FAISSVectorDatabase()
            total_vectors = vector_db.get_count()
            print(f"Total vectors in FAISS: {total_vectors}")

            # Test search
            if manual_entries:
                test_title = manual_entries[0].get('title', '')
                if test_title:
                    print(f"\n🔍 Testing search for: '{test_title}'")
                    results = vector_db.search(test_title, k=3)
                    print(f"Search results: {len(results)}")
                    
                    for i, (doc, score) in enumerate(results):
                        print(f"{i+1}. Score: {score:.3f}")
                        print(f"   Content: {doc[:100]}...")
                        print()

        except Exception as e:
            print(f"❌ Vector database error: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_manual_entries()
