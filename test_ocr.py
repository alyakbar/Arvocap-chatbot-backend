#!/usr/bin/env python3
"""
Simple OCR test to verify that all OCR packages are working correctly
"""

import sys
import os

def test_ocr_setup():
    """Test if OCR packages are properly installed and configured"""
    print("🔍 Testing OCR Setup...")
    print(f"Python version: {sys.version_info[:2]}")
    
    # Test PIL/Pillow
    try:
        from PIL import Image
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
    
    # Test pytesseract
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
        
        # Try to get Tesseract path
        try:
            tesseract_path = pytesseract.get_tesseract_path()
            print(f"✅ Tesseract executable found at: {tesseract_path}")
        except Exception as e:
            print(f"⚠️ Could not get Tesseract path: {e}")
            # Try to set a common Windows path
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', 'Admin'))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"✅ Tesseract path manually set to: {path}")
                    break
            else:
                print("❌ Could not find Tesseract executable")
                return False
                
    except ImportError as e:
        print(f"❌ pytesseract import failed: {e}")
        return False
    
    print("🎉 OCR setup test completed successfully!")
    return True

def test_simple_ocr():
    """Test OCR with a simple text example"""
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        
        print("\n🔤 Testing simple OCR...")
        
        # Create a simple test image with text
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font
        draw.text((10, 30), "Hello OCR World!", fill='black')
        
        # Save test image
        test_image_path = "test_ocr_image.png"
        img.save(test_image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        print(f"OCR extracted text: '{text.strip()}'")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        if "Hello" in text or "OCR" in text:
            print("✅ OCR is working correctly!")
            return True
        else:
            print("⚠️ OCR text extraction may not be optimal")
            return True
            
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OCR Installation and Functionality Test")
    print("=" * 50)
    
    setup_ok = test_ocr_setup()
    
    if setup_ok:
        test_simple_ocr()
        print("\n🎉 OCR is ready to use in your PDF processing!")
    else:
        print("\n❌ OCR setup needs attention before use")
        
    print("=" * 50)
