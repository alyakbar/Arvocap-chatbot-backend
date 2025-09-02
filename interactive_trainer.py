#!/usr/bin/env python3
"""
Interactive Chatbot Training Script
Provides a user-friendly menu to choose training options
"""

import os
import sys
from main import ArvocapTrainingPipeline
from vector_database import ChatbotKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*60)
    print("🤖 ARVOCAP CHATBOT TRAINING SYSTEM")
    print("="*60)
    print("Transform your documents into intelligent conversations")
    print("="*60 + "\n")

def show_menu():
    """Display training options menu"""
    print("📋 TRAINING OPTIONS:")
    print("-" * 30)
    print("1. 📄 Train with PDFs only (OCR supported)")
    print("2. 🌐 Train with Web scraping only")
    print("3. 🔄 Train with both PDFs and Web scraping")
    print("4. 📊 Check current knowledge base status")
    print("5. 🚀 Start API server")
    print("6. 💬 Test chatbot (CLI)")
    print("0. ❌ Exit")
    print("-" * 30)

def check_pdfs():
    """Check if PDF files are available"""
    pdf_dir = "./data/pdfs"
    if not os.path.exists(pdf_dir):
        return 0, []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    return len(pdf_files), pdf_files

def get_knowledge_base_stats():
    """Get current knowledge base statistics"""
    try:
        kb = ChatbotKnowledgeBase()
        size = kb.get_collection_size()
        stats = kb.get_stats()
        return size, stats
    except Exception as e:
        logger.error(f"Error getting KB stats: {e}")
        return 0, {}

def train_pdfs_only():
    """Train with PDFs only"""
    print("\n🚀 Starting PDF-only training...")
    
    # Check available PDFs
    pdf_count, pdf_files = check_pdfs()
    
    if pdf_count == 0:
        print("❌ No PDF files found in ./data/pdfs/")
        print("📁 Please add PDF files to the data/pdfs directory first.")
        return False
    
    print(f"📄 Found {pdf_count} PDF files:")
    for i, pdf in enumerate(pdf_files[:5], 1):  # Show first 5
        print(f"   {i}. {pdf}")
    if pdf_count > 5:
        print(f"   ... and {pdf_count - 5} more files")
    
    # Confirm training
    choice = input(f"\n🔄 Process all {pdf_count} PDFs? (y/n): ").lower().strip()
    if choice != 'y':
        print("❌ Training cancelled.")
        return False
    
    # Initialize and run pipeline
    try:
        pipeline = ArvocapTrainingPipeline()
        success = pipeline.run_full_pipeline(
            include_web=False,
            include_pdfs=True,
            urls=[],
            pdf_dir="./data/pdfs"
        )
        
        if success:
            print("\n✅ PDF training completed successfully!")
            return True
        else:
            print("\n❌ PDF training failed. Check logs above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def train_web_only():
    """Train with web scraping only"""
    print("\n🌐 Starting web scraping training...")
    
    print("📋 Enter URLs to scrape:")
    default_urls = []
    
    for i, url in enumerate(default_urls, 1):
        print(f"   {i}. {url}")
    
    print("\nOptions:")
    print("1. Enter custom URLs")
    print("2. Cancel")
    
    choice = input("Choose option (1-2): ").strip()
    
    urls = []
    if choice == "1":
        print("Enter URLs (one per line, empty line to finish):")
        while True:
            url = input("URL: ").strip()
            if not url:
                break
            if url.startswith(('http://', 'https://')):
                urls.append(url)
            else:
                print("⚠️  Please enter a valid URL starting with http:// or https://")
    else:
        print("❌ Training cancelled.")
        return False
    
    if not urls:
        print("❌ No URLs to scrape. Training cancelled.")
        return False
    
    # Run training
    try:
        pipeline = ArvocapTrainingPipeline()
        success = pipeline.run_full_pipeline(
            include_web=True,
            include_pdfs=False,
            urls=urls,
            pdf_dir=None
        )
        
        if success:
            print("\n✅ Web scraping training completed successfully!")
            return True
        else:
            print("\n❌ Web scraping training failed. Check logs above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def train_both():
    """Train with both PDFs and web scraping"""
    print("\n🔄 Starting comprehensive training (PDFs + Web)...")
    
    # Check PDFs
    pdf_count, pdf_files = check_pdfs()
    
    print(f"📄 Found {pdf_count} PDF files")
    if pdf_count > 0:
        print("   Sample files:", ", ".join(pdf_files[:3]))
        if pdf_count > 3:
            print(f"   ... and {pdf_count - 3} more")
    
    # Configure URLs - get from user input
    print("\n🌐 Web scraping configuration:")
    print("Enter URLs to scrape (one per line, empty line to finish):")
    
    urls = []
    while True:
        url = input("URL: ").strip()
        if not url:
            break
        if url.startswith(('http://', 'https://')):
            urls.append(url)
        else:
            print("⚠️  Please enter a valid URL starting with http:// or https://")
    
    if not urls:
        print("⚠️  No URLs entered. Proceeding with PDF-only training.")
        return train_pdfs_only()
    
    print(f"\n🌐 Will scrape {len(urls)} website(s)")
    for url in urls:
        print(f"   • {url}")
    
    # Confirm
    total_sources = pdf_count + len(urls)
    choice = input(f"\n🚀 Process {total_sources} total sources? (y/n): ").lower().strip()
    if choice != 'y':
        print("❌ Training cancelled.")
        return False
    
    # Run training
    try:
        pipeline = ArvocapTrainingPipeline()
        success = pipeline.run_full_pipeline(
            include_web=True,
            include_pdfs=True,
            urls=urls,
            pdf_dir="./data/pdfs" if pdf_count > 0 else None
        )
        
        if success:
            print("\n✅ Comprehensive training completed successfully!")
            return True
        else:
            print("\n❌ Training failed. Check logs above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

def show_status():
    """Show knowledge base status"""
    print("\n📊 KNOWLEDGE BASE STATUS")
    print("-" * 40)
    
    try:
        size, stats = get_knowledge_base_stats()
        
        print(f"📚 Total documents: {size}")
        if stats:
            print(f"📄 PDF documents: {stats.get('pdf_count', 0)}")
            print(f"🌐 Web documents: {stats.get('web_count', 0)}")
            print(f"📝 Total chunks: {stats.get('total_chunks', 0)}")
        
        # Check PDF directory
        pdf_count, _ = check_pdfs()
        print(f"📁 Available PDFs: {pdf_count}")
        
        if size > 0:
            print("\n✅ Knowledge base is ready for queries!")
        else:
            print("\n⚠️  Knowledge base is empty. Run training first.")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def start_api_server():
    """Start the API server"""
    print("\n🚀 Starting API server...")
    print("The server will start at http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        os.system("python api_server.py")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")

def test_chatbot():
    """Test chatbot in CLI mode"""
    print("\n💬 Starting chatbot CLI...")
    print("Type 'quit' to return to main menu")
    
    try:
        from chatbot_trainer import ChatbotInterface
        
        chatbot = ChatbotInterface(use_openai=True)
        
        print("\n🤖 Chatbot ready! Ask me about your documents.")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("🤖 Bot: ", end="")
            try:
                response = chatbot.generate_response(user_input)
                print(response)
            except Exception as e:
                print(f"Sorry, I encountered an error: {e}")
        
        print("👋 Returning to main menu...")
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")

def main():
    """Main interactive loop"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("🔸 Enter your choice (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using Arvocap Chatbot Training!")
                print("Happy chatting! 🤖")
                break
                
            elif choice == "1":
                train_pdfs_only()
                
            elif choice == "2":
                train_web_only()
                
            elif choice == "3":
                train_both()
                
            elif choice == "4":
                show_status()
                
            elif choice == "5":
                start_api_server()
                
            elif choice == "6":
                test_chatbot()
                
            else:
                print("❌ Invalid choice. Please enter 0-6.")
            
            if choice != "0":
                input("\n⏸️  Press Enter to continue...")
                print("\n" + "="*60 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
