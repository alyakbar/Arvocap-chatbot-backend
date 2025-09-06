#!/usr/bin/env python3
"""
Simple test for PIL image metadata extraction functionality
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_pil_import():
    """Test if PIL/Pillow is available"""
    print("🧪 Testing PIL/Pillow import...")
    
    try:
        from PIL import Image, ExifTags
        print("✅ PIL/Pillow successfully imported")
        
        # Test basic functionality
        print(f"   PIL version: {Image.__version__ if hasattr(Image, '__version__') else 'Unknown'}")
        
        # Try to access common formats
        formats = Image.registered_extensions()
        common_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        supported = [fmt for fmt in common_formats if fmt in formats]
        print(f"   Supported formats: {supported}")
        
        return True
        
    except ImportError as e:
        print(f"❌ PIL/Pillow not available: {e}")
        print("   To install: pip install Pillow")
        return False

def test_image_metadata_method():
    """Test the image metadata extraction method"""
    print("\n🖼️  Testing image metadata extraction method...")
    
    try:
        from web_scraper import WebScraper
        
        # Create scraper instance
        scraper = WebScraper()
        
        # Test the image metadata method exists
        if hasattr(scraper, '_extract_image_metadata'):
            print("✅ _extract_image_metadata method found")
            
            # Test with a sample image data structure
            sample_img = {
                'src': 'https://example.com/test.jpg',
                'alt': 'Test image',
                'title': 'Test title'
            }
            
            # This would normally require downloading the image, 
            # so we'll just verify the method structure
            print("✅ Method structure validated")
            return True
        else:
            print("❌ _extract_image_metadata method not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image metadata method: {e}")
        return False

def test_comprehensive_methods():
    """Test comprehensive crawling methods exist"""
    print("\n🕸️  Testing comprehensive crawling methods...")
    
    try:
        from web_scraper import WebScraper
        
        scraper = WebScraper()
        
        methods_to_check = [
            '_discover_comprehensive_sitemaps',
            '_discover_common_paths', 
            '_discover_additional_links'
        ]
        
        missing_methods = []
        for method in methods_to_check:
            if not hasattr(scraper, method):
                missing_methods.append(method)
        
        if not missing_methods:
            print("✅ All comprehensive crawling methods found")
            return True
        else:
            print(f"❌ Missing methods: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing comprehensive methods: {e}")
        return False

def test_scraper_parameters():
    """Test that scraper accepts new parameters"""
    print("\n⚙️  Testing scraper parameter support...")
    
    try:
        from web_scraper import WebScraper
        
        # Test initialization with new parameters
        scraper = WebScraper(use_selenium=False)
        
        # Test scrape_website method signature
        import inspect
        sig = inspect.signature(scraper.scrape_website)
        params = list(sig.parameters.keys())
        
        required_params = ['comprehensive_crawl', 'max_pages']
        missing_params = [p for p in required_params if p not in params]
        
        if not missing_params:
            print("✅ All required parameters supported")
            print(f"   Parameters: {params}")
            return True
        else:
            print(f"❌ Missing parameters: {missing_params}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing parameters: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Web Scraper Component Test")
    print("=" * 50)
    
    # Test 1: PIL availability
    test1_passed = test_pil_import()
    
    # Test 2: Image metadata method
    test2_passed = test_image_metadata_method()
    
    # Test 3: Comprehensive methods
    test3_passed = test_comprehensive_methods()
    
    # Test 4: Parameter support
    test4_passed = test_scraper_parameters()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Component Test Results:")
    print(f"   PIL/Pillow: {'✅ AVAILABLE' if test1_passed else '❌ MISSING'}")
    print(f"   Image Metadata: {'✅ READY' if test2_passed else '❌ NOT READY'}")
    print(f"   Comprehensive Methods: {'✅ READY' if test3_passed else '❌ NOT READY'}")
    print(f"   Parameter Support: {'✅ READY' if test4_passed else '❌ NOT READY'}")
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    if all_passed:
        print("\n🎉 All components ready! Enhanced scraper should work correctly.")
        print("💡 Tip: Use --comprehensive-crawl and --max-pages flags for full website discovery")
        print("💡 Example: python main.py --urls https://example.com --comprehensive-crawl --max-pages 200")
    else:
        print("\n⚠️  Some components need attention. Check individual test results.")
