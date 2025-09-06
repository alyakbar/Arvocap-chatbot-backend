#!/usr/bin/env python3
"""
Test manual entry functionality
"""

import requests
import json

def test_manual_entry():
    print("🧪 TESTING MANUAL ENTRY")
    print("=" * 30)
    
    # Test data
    test_data = {
        "title": "Test Manual Entry",
        "content": "This is a test manual entry to verify the system works properly. It contains information about testing.",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    try:
        # Test API server health first
        print("🔍 Checking API server health...")
        health_response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API server is {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ API server health check failed: {health_response.status_code}")
            return False
        
        # Test manual entry endpoint
        print("📝 Testing manual entry endpoint...")
        response = requests.post(
            "http://127.0.0.1:8000/admin/add_manual_entry",
            json=test_data,
            timeout=10
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Manual entry added successfully!")
            print(f"   Success: {result.get('success')}")
            print(f"   Chunks created: {result.get('chunks_created')}")
            print(f"   Message: {result.get('message')}")
            return True
        else:
            print(f"❌ Manual entry failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_manual_entry():
    """Verify the manual entry was stored"""
    print("\n🔍 VERIFYING STORAGE")
    print("=" * 25)
    
    try:
        from vector_database import ChatbotKnowledgeBase
        
        kb = ChatbotKnowledgeBase()
        items = kb.get_all_items()
        
        # Look for manual entries
        manual_entries = [item for item in items if item.get('source_type') == 'manual']
        print(f"Manual entries found: {len(manual_entries)}")
        
        if manual_entries:
            latest = manual_entries[-1]
            print(f"Latest manual entry:")
            print(f"   Title: {latest.get('title', 'No title')}")
            print(f"   Content: {latest.get('content', '')[:100]}...")
            print(f"   Source: {latest.get('source', 'Unknown')}")
            return True
        else:
            print("❌ No manual entries found in knowledge base")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying storage: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Manual Entry Test Suite")
    print("=" * 40)
    
    # Test adding manual entry
    add_success = test_manual_entry()
    
    if add_success:
        # Verify it was stored
        verify_success = verify_manual_entry()
        
        if verify_success:
            print("\n🎉 Manual entry system is working correctly!")
        else:
            print("\n⚠️ Manual entry added but verification failed")
    else:
        print("\n❌ Manual entry system has issues")
