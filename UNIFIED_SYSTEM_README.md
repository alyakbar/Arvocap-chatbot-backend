# 🚀 Unified Arvocap Chatbot System

A complete AI-powered chatbot system that combines web scraping, PDF processing, and intelligent chat responses in one unified application.

## ✨ Features

- 🌐 **Web Scraping**: Automatically extract content from websites
- 📄 **PDF Processing**: Extract text from PDFs with OCR support for scanned documents
- 🤖 **AI Chat**: Intelligent responses using OpenAI GPT models
- 🔍 **Vector Search**: Semantic search through your knowledge base
- 📊 **Real-time Training**: Add new content and retrain on-the-fly
- 🎯 **Unified API**: Single endpoint for all operations

## 🚀 Quick Start

### Option 1: Windows Batch Script (Easiest)
```bash
# Double-click this file:
start_unified_system.bat
```

### Option 2: Command Line
```bash
# Start the server
python unified_chatbot_system.py --mode server

# Or run training
python unified_chatbot_system.py --mode train --urls https://example.com
```

### Option 3: Direct Python
```python
# Import and use directly
from unified_chatbot_system import UnifiedChatbotSystem

system = UnifiedChatbotSystem()
# Use the system programmatically
```

## 🌐 API Endpoints

Once running, access the system at `http://localhost:8000`

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System overview and status |
| `/docs` | GET | Interactive API documentation |
| `/chat` | POST | Chat with the AI |
| `/train` | POST | Run training pipeline |
| `/scrape` | POST | Scrape websites |
| `/status` | GET | Get system status |
| `/upload-pdf` | POST | Upload and process PDF |

### Example Usage

#### 1. Chat with the AI
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What investment funds do you have information about?"}'
```

#### 2. Scrape Websites
```bash
curl -X POST "http://localhost:8000/scrape" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://www.arvocap.com", "https://www.cma.or.ke"],
    "max_pages": 10
  }'
```

#### 3. Run Complete Training
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "include_web": true,
    "include_pdfs": true,
    "clear_existing": false
  }'
```

#### 4. Upload PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@your_document.pdf"
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_key (optional)
```

### Data Directories
```
python_training/
├── data/
│   ├── pdfs/           # Place PDF files here
│   ├── uploads/        # Uploaded files go here
│   └── processed/      # Processed data storage
├── vector_db/          # Vector database storage
└── logs/              # System logs
```

## 💻 Command Line Interface

### Server Mode (Default)
```bash
python unified_chatbot_system.py --mode server --host 0.0.0.0 --port 8000
```

### Training Mode
```bash
# Train with web scraping and PDFs
python unified_chatbot_system.py --mode train --urls https://example.com

# Train only PDFs
python unified_chatbot_system.py --mode train --no-web

# Clear existing data and retrain
python unified_chatbot_system.py --mode train --clear
```

### Scraping Mode
```bash
python unified_chatbot_system.py --mode scrape --urls https://site1.com https://site2.com
```

### Status Check
```bash
python unified_chatbot_system.py --mode status
```

## 🔍 Web Scraping Capabilities

The system can extract content from:
- ✅ Static HTML websites
- ✅ JavaScript-rendered pages (with Selenium)
- ✅ Multi-page crawling
- ✅ Financial websites and reports
- ✅ News articles and blogs
- ✅ Documentation sites

### Supported Content Types
- Text content and articles
- Tables and structured data
- Navigation menus and links
- Meta descriptions and titles

## 📄 PDF Processing Features

Advanced PDF processing with multiple extraction methods:
- ✅ **Text-based PDFs**: Fast extraction using PyMuPDF/pdfplumber
- ✅ **Scanned PDFs**: OCR extraction using Tesseract
- ✅ **Mixed content**: Intelligent method selection
- ✅ **Financial documents**: Optimized for fund reports, statements
- ✅ **Table extraction**: Converts tables to readable text

### Supported PDF Types
- Investment fund reports and fact sheets
- Financial statements and analysis
- Research documents and whitepapers
- Regulatory filings and compliance docs
- Marketing materials and brochures

## 🤖 AI Chat Features

Intelligent responses powered by:
- **OpenAI GPT Models**: Latest GPT-3.5/GPT-4 integration
- **Vector Search**: Semantic similarity matching
- **Context Awareness**: Relevant information retrieval
- **Financial Focus**: Optimized for investment/finance queries

### Sample Queries
- "What investment funds are available?"
- "Tell me about fund performance in 2024"
- "What are the fees for the money market fund?"
- "Explain the risk factors for equity investments"
- "How do I invest in the balanced fund?"

## 📊 System Monitoring

### Real-time Status
- Knowledge base document count
- Last training/update timestamps
- Component availability status
- Training progress indicators

### Performance Metrics
- Response times for chat queries
- Scraping success rates
- PDF processing statistics
- Vector search accuracy

## 🛠️ Advanced Usage

### Programmatic Access
```python
from unified_chatbot_system import UnifiedChatbotSystem

# Initialize system
system = UnifiedChatbotSystem()

# Scrape websites
result = await system.scrape_websites(['https://example.com'])

# Process PDFs
pdf_result = await system.process_pdfs('/path/to/pdfs')

# Update knowledge base
await system.update_knowledge_base(result, pdf_result)

# Chat with AI
response = system.chatbot.generate_response("Your question here")
```

### Batch Processing
```python
# Process multiple sources at once
await system.full_training_pipeline(
    urls=['https://site1.com', 'https://site2.com'],
    include_web=True,
    include_pdfs=True,
    clear_existing=False
)
```

## 🔒 Security Considerations

- API endpoints can be secured with authentication
- File uploads are validated and sandboxed
- Web scraping respects robots.txt and rate limits
- Sensitive data can be filtered during processing

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **OCR Not Working**
   - Install Tesseract: `winget install tesseract`
   - Check OCR libraries: `pip install pytesseract pdf2image`

3. **Web Scraping Fails**
   - Check internet connection
   - Some sites may block automated requests
   - Try with Selenium for JS-heavy sites

4. **Large PDF Processing**
   - OCR processing can be slow for large files
   - Consider processing smaller batches
   - Monitor memory usage

### Performance Optimization

- **Concurrent Processing**: System uses async/await for better performance
- **Chunking Strategy**: Optimized text chunking for better search results  
- **Caching**: Vector embeddings are cached for faster retrieval
- **Background Tasks**: Heavy processing runs in background

## 📈 Scaling Considerations

- **Database**: Can be upgraded to PostgreSQL for production
- **Caching**: Redis can be added for session management
- **Load Balancing**: Multiple instances can run behind a load balancer
- **Storage**: File storage can be moved to cloud (S3, etc.)

## 🤝 Integration Examples

### Frontend Integration
```javascript
// Chat with the system
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello!' })
});
```

### Webhook Integration
```python
# Trigger training when new content is available
requests.post('http://localhost:8000/train', json={
  'urls': ['https://new-content.com'],
  'include_web': True,
  'include_pdfs': True
})
```

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB storage space
- Internet connection

### Recommended Requirements  
- Python 3.11+
- 8GB RAM
- 10GB storage space
- High-speed internet
- GPU for faster OCR processing (optional)

## 🎯 Use Cases

### Investment Management
- Process fund reports and performance data
- Extract regulatory filing information
- Analyze market research documents
- Answer client investment questions

### Financial Advisory
- Create knowledge base from research materials
- Process client documents and statements
- Provide instant answers to common questions
- Generate insights from multiple sources

### Document Processing
- Batch process large PDF collections
- Extract structured data from reports
- Convert legacy documents to searchable format
- Create unified knowledge repositories

---

## 🎉 Success! Your Unified System is Ready

The Unified Arvocap Chatbot System combines all the individual components into one powerful application that can:

1. ✅ **Scrape any website** and extract valuable content
2. ✅ **Process any PDF** (including scanned documents with OCR)
3. ✅ **Train the AI** with new knowledge automatically
4. ✅ **Serve intelligent responses** through a modern API
5. ✅ **Scale and adapt** to your growing knowledge needs

**Start the system and begin building your AI-powered knowledge base today!** 🚀
