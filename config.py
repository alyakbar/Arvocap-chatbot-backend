import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# Web Scraping Settings
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
REQUEST_DELAY = 1  # Delay between requests in seconds
MAX_RETRIES = 3

# Text Processing Settings
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Model Settings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHAT_MODEL = 'gpt-3.5-turbo'

# Vector Database Settings
# Use an absolute path to ensure consistency
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vector_db'))
COLLECTION_NAME = 'chatbot_knowledge'

# Training Settings
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512

# File Paths
TRAINING_DATA_PATH = '../data/training-data.json'
SCRAPED_DATA_PATH = './data/scraped_data.json'
MODEL_OUTPUT_PATH = './models/'
LOGS_PATH = './logs/'
