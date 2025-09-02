# 🤖 Arvocap Chatbot Training System

A comprehensive AI-powered chatbot training and serving system that combines web scraping, PDF processing (including OCR), and intelligent chat responses using OpenAI GPT models and vector databases.

## 🌟 Features

### 📊 **Intelligent Data Processing**
- **Web Scraping**: Extract content from websites using both standard HTTP requests and Selenium for JavaScript-heavy sites
- **PDF Processing**: Handle both text-based and scanned PDFs with OCR support using Tesseract
- **Text Analysis**: Clean, chunk, and analyze text content with keyword extraction and Q&A pair generation
- **Vector Database**: Store and search through knowledge using ChromaDB with semantic similarity

### 🤖 **AI Chat Capabilities**
- **OpenAI Integration**: Powered by GPT-3.5/GPT-4 for intelligent responses
- **Context-Aware**: Retrieves relevant information from your knowledge base
- **Real-time API**: FastAPI server for seamless integration with web applications
- **Conversation Management**: Support for conversation tracking and context

### 🔧 **Training & Management**
- **Automated Training**: Complete pipeline from data collection to chatbot deployment
- **Knowledge Base Updates**: Add new content and retrain on-the-fly
- **Performance Monitoring**: Track document counts, response times, and system health
- **Background Processing**: Handle large datasets without blocking operations

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (tested with Python 3.13)
- **OpenAI API Key** (required for best results)
- **Tesseract OCR** (automatically installed via winget)

### 1. Environment Setup
```bash
# Navigate to the python_training directory
cd python_training

# Activate the virtual environment
.\env\Scripts\activate  # Windows
# or
source env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the `python_training` directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_optional
```

### 3. Start the API Server
```bash
# Simple API server (recommended for Next.js integration)
python api_server.py

# Or unified system (includes all features)
python unified_chatbot_system.py
```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
python_training/
├── 🔧 Core Components
│   ├── api_server.py              # Main FastAPI server for Next.js integration
│   ├── unified_chatbot_system.py  # Complete unified system with all features
│   ├── main.py                    # Training pipeline orchestrator
│   └── config.py                  # Central configuration
│
├── 📄 Data Processing
│   ├── web_scraper.py             # Website content extraction
│   ├── pdf_processor.py           # PDF processing with OCR support
│   ├── text_processor.py          # Text cleaning and analysis
│   └── vector_database.py         # ChromaDB vector storage
│
├── 🤖 AI & Training
│   ├── chatbot_trainer.py         # Model training and chat interface
│   └── test_ocr.py               # OCR functionality testing
│
├── 📊 Data & Storage
│   ├── data/                     # Training data and processed files
│   ├── vector_db/               # Vector database storage
│   ├── models/                  # Model storage
│   └── logs/                    # System logs
│
├── ⚙️ Configuration
│   ├── requirements.txt         # Python dependencies
│   ├── .env                    # API keys and secrets
│   └── env/                    # Virtual environment
│
└── 📚 Documentation
    ├── README.md                # This file
    ├── UNIFIED_SYSTEM_README.md # Detailed unified system guide
    ├── PDF_TRAINING_README.md   # PDF processing documentation
    └── OCR_TRAINING_GUIDE.md    # OCR setup and usage guide
```

## 🛠️ Usage Examples

### API Integration (Next.js)
```javascript
// Chat with the AI
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    message: 'What investment funds do you have information about?' 
  })
});

// Search the knowledge base
const searchResults = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    query: 'investment performance',
    max_results: 5 
  })
});
```

### Training Pipeline
```python
from main import ArvocapTrainingPipeline

# Initialize pipeline
pipeline = ArvocapTrainingPipeline()

# Train with web content
urls = ["https://www.example.com", "https://www.finance-site.com"]
pipeline.collect_web_data(urls)

# Process PDFs (including scanned documents)
pipeline.collect_pdf_data("./data/pdfs")

# Build knowledge base
pipeline.process_and_train()
```

### Direct API Usage
```python
from chatbot_trainer import ChatbotInterface
from vector_database import ChatbotKnowledgeBase

# Initialize chatbot
chatbot = ChatbotInterface(use_openai=True)
kb = ChatbotKnowledgeBase()

# Add documents to knowledge base
kb.add_document("Your financial content here", {"source": "manual"})

# Generate response
response = chatbot.generate_response("Tell me about investment options")
```

## 🌐 API Endpoints

### Core Chat & Search
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System overview and health check |
| `/chat` | POST | Chat with the AI chatbot |
| `/search` | POST | Vector search through knowledge base |
| `/health` | GET | Detailed system health information |

### Knowledge Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/knowledge/stats` | GET | Knowledge base statistics |
| `/knowledge/search` | POST | Search knowledge base content |
| `/scrape` | POST | Scrape websites and add to knowledge base |
| `/upload-pdf` | POST | Upload and process PDF documents |

### Training & Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Run complete training pipeline |
| `/status` | GET | Get training and system status |
| `/docs` | GET | Interactive API documentation (Swagger) |

## 🔍 Advanced Features

### Web Scraping Capabilities
- **Multi-format Support**: HTML, JavaScript-rendered pages, dynamic content
- **Intelligent Crawling**: Respects robots.txt and implements rate limiting
- **Content Extraction**: Clean text extraction with metadata preservation
- **Link Following**: Configurable depth crawling within domains

### PDF Processing & OCR
- **Multiple Extraction Methods**: PyMuPDF, pdfplumber, pypdf for text-based PDFs
- **OCR Support**: Tesseract integration for scanned documents
- **Smart Processing**: Automatically detects best extraction method
- **Financial Documents**: Optimized for fund reports, statements, compliance docs

### Vector Database Features
- **Semantic Search**: Find relevant content using meaning, not just keywords
- **Efficient Storage**: ChromaDB for fast similarity searches
- **Metadata Support**: Rich metadata for filtering and organization
- **Scalable**: Handles thousands of documents efficiently

### AI Chat Features
- **Context Retrieval**: Automatically finds relevant background information
- **Source Attribution**: Shows which documents informed the response
- **Conversation Memory**: Maintains context across chat sessions
- **Customizable**: Adjust response style and behavior

## 🔧 Configuration Options

### Model Settings (`config.py`)
```python
# AI Model Configuration
CHAT_MODEL = 'gpt-3.5-turbo'  # or 'gpt-4'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Text Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TEXT_LENGTH = 5000

# Vector Database
COLLECTION_NAME = 'chatbot_knowledge'
```

### Environment Variables (`.env`)
```env
# Required
OPENAI_API_KEY=your_key_here

# Optional
HUGGINGFACE_API_KEY=your_key_here
```

## 🚨 Troubleshooting

### Common Issues

**1. OCR Not Working**
```bash
# Install Tesseract OCR
winget install --id UB-Mannheim.TesseractOCR

# Verify OCR packages
python test_ocr.py
```

**2. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. API Server Won't Start**
```bash
# Check if port is in use
netstat -an | findstr :8000

# Try different port
python api_server.py --port 8001
```

**4. Memory Issues**
- Reduce `CHUNK_SIZE` in config.py for large documents
- Process PDFs in smaller batches
- Use `max_pages` parameter for web scraping

### Performance Optimization

**For Large Datasets:**
- Use background processing for training
- Implement pagination for API responses
- Consider upgrading to PostgreSQL for production

**For Better Responses:**
- Add more diverse training data
- Increase context window size
- Use GPT-4 for better reasoning

## 🎯 Use Cases

### Financial Advisory
- **Investment Research**: Process fund reports and market analysis
- **Client Documentation**: Extract information from statements and filings
- **Regulatory Compliance**: Search through compliance documents
- **Market Intelligence**: Monitor financial news and reports

### Document Management
- **Knowledge Base Creation**: Convert document collections to searchable databases
- **Content Migration**: Extract content from legacy PDF systems
- **Research Assistance**: Find specific information across large document sets
- **Automated Summarization**: Generate summaries from multiple sources

### Customer Support
- **FAQ Automation**: Answer common questions using existing documentation
- **Product Information**: Provide detailed product specifications and features
- **Policy Explanation**: Help customers understand complex policies and procedures
- **Troubleshooting**: Guide users through problem resolution steps

---

## 📞 Support & Contributing

### Getting Help
- **Documentation**: Check the guides in the project folder
- **Issues**: Report bugs and request features on GitHub
- **Community**: Contact support for discussions and help

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the Arvocap community**

*Transform your documents into intelligent conversations*
