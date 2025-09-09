# Docker Management Commands

## Build and Start Services
```bash
# Build and start containers in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f
```

## Development Commands
```bash
# Start development environment
docker-compose up

# Run tests
docker-compose exec api pytest

# Check logs
docker-compose exec api tail -f /app/logs/app.log

# Enter container shell
docker-compose exec api bash
```

## Database Management
```bash
# Backup vector database
docker-compose exec api python backup_vector_db.py

# Rebuild vector database
docker-compose exec api python rebuild_vector_db.py
```

## Maintenance
```bash
# Stop all containers
docker-compose down

# Remove volumes (careful - this deletes data!)
docker-compose down -v

# Update containers
docker-compose pull
docker-compose up -d --build
```

## Environment Variables
Required environment variables:
- OPENAI_API_KEY: Your OpenAI API key
- GOOGLE_SHEETS_ID: Google Sheets ID for contact form

Optional environment variables:
- CHAT_CONCURRENCY: Number of concurrent chat requests (default: 8)
- CHUNK_SIZE: Text chunk size for processing (default: 1000)
- CHUNK_OVERLAP: Overlap between chunks (default: 200)
- DEBUG: Enable debug mode (0/1)
