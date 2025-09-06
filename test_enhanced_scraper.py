#!/usr/bin/env python3
"""
Test script for enhanced web scraper with image metadata extraction
"""

import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from web_scraper import WebScraper

def test_image_metadata_extraction():
    """Test image metadata extraction on a simple webpage"""
    print("🧪 Testing Enhanced Web Scraper with Image Metadata...")
    
    try:
        # Initialize scraper
        scraper = WebScraper(use_selenium=False)
        
        # Test with a website that has images
        test_url = "https://www.arvocap.com/about-us"  # Simple test page
        
        print(f"📄 Scraping: {test_url}")
        
        pages = scraper.scrape_website(
            test_url,
            max_pages=1,  # Just test one page
            same_domain_only=True,
            max_depth=1,
            use_sitemaps=False,
            respect_robots=True,
            comprehensive_crawl=False,  # Simple test first
        )
        
        if pages:
            page = pages[0]
            print(f"✅ Successfully scraped page: {page.get('url', 'Unknown')}")
            print(f"📝 Title: {page.get('title', 'No title')}")
            print(f"📄 Content length: {len(page.get('content', ''))}")
            
            # Check for images
            images = page.get('images', [])
            print(f"🖼️  Found {len(images)} images")
            
            for i, img in enumerate(images):
                print(f"\n📸 Image {i+1}:")
                print(f"   URL: {img.get('src', 'No src')}")
                print(f"   Alt text: {img.get('alt', 'No alt')}")
                print(f"   Title: {img.get('title', 'No title')}")
                if 'dimensions' in img:
                    dims = img['dimensions']
                    print(f"   Dimensions: {dims.get('width', '?')}x{dims.get('height', '?')}")
                if 'file_info' in img:
                    info = img['file_info']
                    print(f"   Format: {info.get('format', 'Unknown')}")
                    print(f"   Size: {info.get('size', 'Unknown')} bytes")
                if 'context' in img:
                    ctx = img['context']
                    print(f"   Caption: {ctx.get('caption', 'No caption')}")
                    if ctx.get('surrounding_text'):
                        text = ctx['surrounding_text'][:100] + "..." if len(ctx['surrounding_text']) > 100 else ctx['surrounding_text']
                        print(f"   Context: {text}")
            
            return True
        else:
            print("❌ No pages scraped")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_crawling():
    """Test comprehensive crawling features"""
    print("\n🕸️  Testing Comprehensive Crawling...")
    
    try:
        scraper = WebScraper(use_selenium=False)
        
        # Test with a simple site
        test_url = "https://www.arvocap.com/about-us"
        
        print(f"🌐 Testing comprehensive crawl: {test_url}")
        
        pages = scraper.scrape_website(
            test_url,
            max_pages=5,
            same_domain_only=True,
            max_depth=2,
            use_sitemaps=True,
            respect_robots=True,
            comprehensive_crawl=True,
        )
        
        print(f"✅ Comprehensive crawl found {len(pages)} pages")
        
        for i, page in enumerate(pages):
            print(f"   Page {i+1}: {page.get('url', 'Unknown')}")
            if page.get('title'):
                print(f"      Title: {page['title'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive crawl test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Web Scraper Test Suite")
    print("=" * 50)
    
    # Test 1: Image metadata extraction
    test1_passed = test_image_metadata_extraction()
    
    # Test 2: Comprehensive crawling
    test2_passed = test_comprehensive_crawling()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Image Metadata: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Comprehensive Crawl: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Enhanced scraper is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
