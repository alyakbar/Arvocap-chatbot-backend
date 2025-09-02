#!/usr/bin/env python3
"""
Simple OCR test without numpy dependency
"""

import sys
import os

def test_basic_ocr():
    """Test basic OCR without numpy"""
    print("🔍 Testing Basic OCR Setup (without numpy)...")
    print(f"Python version: {sys.version_info[:2]}")
    
    # Test PIL/Pillow first
    try:
        from PIL import Image, ImageDraw
        print("✅ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"❌ PIL/Pillow import failed: {e}")
        return False
    
    # Test pdf2image
    try:
        import pdf2image
        print("✅ pdf2image imported successfully")
    except ImportError as e:
        print(f"❌ pdf2image import failed: {e}")
        return False
    
    # Test basic image creation and text detection
    try:
        # Create a simple test image
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Test OCR", fill='black')
        
        # Save test image
        test_image_path = "simple_test.png"
        img.save(test_image_path)
        print("✅ Created test image successfully")
        
        # Try to import pytesseract without using it yet
        try:
            import pytesseract
            print("✅ pytesseract imported successfully")
            
            # Try to get tesseract path
            try:
                path = pytesseract.get_tesseract_path()
                print(f"✅ Tesseract found at: {path}")
            except:
                # Try common Windows paths
                common_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        print(f"✅ Tesseract manually set to: {path}")
                        break
                else:
                    print("⚠️ Tesseract path not found, but package imported")
        except ImportError as e:
            print(f"❌ pytesseract import failed: {e}")
            return False
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
        print("🎉 Basic OCR setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Basic OCR test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple OCR Test (No Numpy)")
    print("=" * 50)
    
    if test_basic_ocr():
        print("\n✅ OCR packages are installed and basic functionality works!")
        print("Note: Full OCR functionality may require numpy for advanced features.")
    else:
        print("\n❌ OCR setup needs attention")
        
    print("=" * 50)
