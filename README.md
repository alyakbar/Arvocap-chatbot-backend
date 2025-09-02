# Chatbot Training System

This Python-based system allows you to create and train a chatbot by scraping data from websites.

## Features

- **Web Scraping**: Extract content from websites (supports both regular and JavaScript-heavy sites)
- **Text Processing**: Clean, chunk, and analyze scraped text
- **Vector Database**: Store and search through knowledge using ChromaDB
- **Training Options**: 
  - OpenAI fine-tuning
  - Local model training with Transformers
- **Interactive Chat**: Test your trained chatbot

## Installation

1. **Install Python 3.8+**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Setup environment variables:**
   - Copy `.env` file and add your API keys
   - At minimum, add your OpenAI API key for best results

## Quick Start

1. **Run the main pipeline:**
   ```bash
   python main.py
   ```

2. **Choose option 1 for full pipeline**

3. **Enter a website URL to scrape**

4. **Follow the prompts to train your chatbot**

## Individual Components

### Web Scraper (`web_scraper.py`)
```bash
python web_scraper.py
```
- Scrapes websites and extracts text content
- Supports both requests-based and Selenium-based scraping
- Handles multiple pages and follows links

### Text Processor (`text_processor.py`)
```bash
python text_processor.py
```
- Cleans and processes scraped text
- Extracts keywords and Q&A pairs
- Chunks content for better processing
- Clusters similar content

### Vector Database (`vector_database.py`)
```bash
python vector_database.py
```
- Creates searchable knowledge base
- Uses sentence embeddings for similarity search
- Stores documents and Q&A pairs

### Chatbot Trainer (`chatbot_trainer.py`)
```bash
python chatbot_trainer.py
```
- Prepares training data
- Supports OpenAI fine-tuning
- Can train local models with Transformers
- Provides chat interface

## Usage Examples

### 1. Scrape a Company Website
```python
from web_scraper import WebScraper

scraper = WebScraper()
data = scraper.scrape_website("https://example.com", max_pages=20)
```

### 2. Process Text Data
```python
from text_processor import TextProcessor

processor = TextProcessor()
processed = processor.process_scraped_data(scraped_data)
```

### 3. Build Knowledge Base
```python
from vector_database import ChatbotKnowledgeBase

kb = ChatbotKnowledgeBase()
kb.load_training_data('processed_data.json')
```

### 4. Train and Chat
```python
from chatbot_trainer import ChatbotInterface

chatbot = ChatbotInterface(use_openai=True)
response = chatbot.generate_response("Hello!")
```

## Configuration

Edit `config.py` to customize:
- Model settings
- Text processing parameters
- API endpoints
- File paths

## File Structure

```
python_training/
├── main.py              # Main pipeline script
├── config.py            # Configuration settings
├── web_scraper.py       # Website scraping
├── text_processor.py    # Text processing and analysis
├── vector_database.py   # Vector database operations
├── chatbot_trainer.py   # Model training and chat interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
└── data/               # Generated data files
    ├── scraped_data.json
    ├── processed_data.json
    └── openai_training.jsonl
```

## Training Process

1. **Scraping**: Extract content from target website
2. **Processing**: Clean text, extract keywords, create Q&A pairs
3. **Knowledge Base**: Store in vector database for similarity search
4. **Training**: 
   - OpenAI: Create fine-tuning dataset
   - Local: Train with Transformers library
5. **Testing**: Interactive chat interface

## Tips for Best Results

1. **Website Selection**: Choose content-rich websites with clear structure
2. **API Keys**: Use OpenAI API for best results
3. **Data Quality**: Review scraped content for relevance
4. **Training Data**: More diverse, high-quality data = better chatbot
5. **Testing**: Test with various question types

## Troubleshooting

### Common Issues:

1. **Selenium WebDriver**: Install ChromeDriver for JavaScript sites
2. **spaCy Model**: Run `python -m spacy download en_core_web_sm`
3. **Memory Issues**: Reduce batch size or max pages
4. **API Limits**: Check OpenAI API quotas and billing

### Error Handling:

The system includes comprehensive error handling and logging. Check the console output for detailed error messages.

## Integration with Next.js App

The training system can integrate with your existing Next.js chatbot:

1. Export training data to JSON format
2. Use the trained model via API
3. Import knowledge base into your existing system
4. Update your chatbot's response generation

## Advanced Features

- **Custom Models**: Support for different transformer models
- **Incremental Training**: Add new data without retraining from scratch
- **Multi-language**: Extend for non-English content
- **Custom Embeddings**: Use domain-specific embedding models

## Contributing

Feel free to extend the system with:
- Additional data sources
- New training algorithms
- Better text processing
- Enhanced chat interfaces
