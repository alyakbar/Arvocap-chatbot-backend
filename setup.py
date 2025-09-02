#!/usr/bin/env python3
"""
Setup script for the Chatbot Training System
Installs dependencies and configures the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def install_requirements():
    """Install Python packages from requirements.txt"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def download_spacy_model():
    """Download spaCy English model"""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )

def download_nltk_data():
    """Download required NLTK data"""
    script = '''
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
'''
    
    return run_command(
        f"{sys.executable} -c \"{script}\"",
        "Downloading NLTK data"
    )

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['data', 'logs', 'models', 'vector_db']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created")
    return True

def setup_environment_file():
    """Setup environment file with instructions"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("📝 Creating .env file...")
    
    env_content = """# Environment variables for Chatbot Training System
# Add your API keys here

# OpenAI API Key (required for best results)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face API Key (optional)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Google API Key (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Instructions:
# 1. Get your OpenAI API key from: https://platform.openai.com/api-keys
# 2. Replace 'your_openai_api_key_here' with your actual API key
# 3. Remove this comment section after setup
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
        print("🔑 Please edit .env file and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    test_script = '''
import sys
import traceback

packages = [
    "requests",
    "beautifulsoup4", 
    "selenium",
    "pandas",
    "numpy",
    "nltk",
    "spacy",
    "sklearn",
    "transformers",
    "torch",
    "sentence_transformers",
    "chromadb",
    "openai",
    "python-dotenv"
]

failed_imports = []

for package in packages:
    try:
        if package == "beautifulsoup4":
            import bs4
        elif package == "python-dotenv":
            import dotenv
        elif package == "sentence_transformers":
            import sentence_transformers
        elif package == "sklearn":
            import sklearn
        else:
            __import__(package)
        print(f"✅ {package}")
    except ImportError as e:
        print(f"❌ {package}: {e}")
        failed_imports.append(package)
    except Exception as e:
        print(f"⚠️  {package}: {e}")

if failed_imports:
    print(f"\\nFailed imports: {failed_imports}")
    sys.exit(1)
else:
    print("\\n🎉 All packages imported successfully!")
'''
    
    return run_command(
        f"{sys.executable} -c \"{test_script}\"",
        "Testing package imports"
    )

def main():
    """Main setup function"""
    print("🚀 Chatbot Training System Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    if success and not create_directories():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Download spaCy model
    if success and not download_spacy_model():
        print("⚠️  spaCy model download failed, but continuing...")
    
    # Download NLTK data
    if success and not download_nltk_data():
        print("⚠️  NLTK data download failed, but continuing...")
    
    # Setup environment file
    if success and not setup_environment_file():
        success = False
    
    # Test imports
    if success and not test_imports():
        success = False
    
    print("\n" + "=" * 40)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: python main.py")
        print("3. Choose option 1 for full pipeline")
        print("4. Enter a website URL to scrape")
    else:
        print("❌ Setup failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("- Ensure Python 3.8+ is installed")
        print("- Check internet connection")
        print("- Try running: pip install --upgrade pip")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)
