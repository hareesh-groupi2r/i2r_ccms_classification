# 🚀 Production Deployment Guide

## Contract Correspondence Multi-Category Classification System

This guide covers deploying the production-ready API for the Contract Correspondence Classification System.

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose (optional)
- 4GB+ RAM recommended
- OpenAI API key (optional, for LLM classification)

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies

```bash
# Install production requirements
pip install -r requirements_production.txt

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_production.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-your-anthropic-key-here
ENVIRONMENT=development
```

### 3. Start the API

```bash
# Using the startup script (recommended)
python start_production.py

# Or directly with uvicorn
uvicorn production_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

```bash
# Run comprehensive tests
python test_production_api.py

# Check health
curl http://localhost:8000/health

# View documentation
open http://localhost:8000/docs
```

## 🐳 Docker Deployment

### Quick Docker Run

```bash
# Build the image
docker build -t ccms-api .

# Run the container
docker run -d \
  --name ccms-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env:ro \
  ccms-api
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ccms-api

# Scale API instances
docker-compose up -d --scale ccms-api=3

# Stop all services
docker-compose down
```

## 🏢 Production Deployment

### Environment Configuration

```bash
# Set production environment variables
export ENVIRONMENT=production
export WORKERS=4
export PORT=8000
export OPENAI_API_KEY=sk-your-key
```

### Using Gunicorn (Production WSGI Server)

```bash
# Install gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn production_api:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --keep-alive 2 \
  --max-requests 1000 \
  --max-requests-jitter 100
```

### Systemd Service (Linux)

Create `/etc/systemd/system/ccms-api.service`:

```ini
[Unit]
Description=Contract Correspondence Classification API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/ccms-classification
Environment=PATH=/opt/ccms-classification/venv/bin
ExecStart=/opt/ccms-classification/venv/bin/gunicorn production_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ccms-api
sudo systemctl start ccms-api
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | deployment environment | `development` |
| `WORKERS` | Number of worker processes | `4` |
| `PORT` | API port | `8000` |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |

### API Configuration

Edit `production_config.yaml` for advanced settings:

```yaml
api:
  workers: 4
  max_requests_per_worker: 1000
  timeout: 300
  
approaches:
  pure_llm:
    enabled: true
    model: "gpt-4-turbo"
  hybrid_rag:
    enabled: true
    top_k: 15
```

## 📊 Monitoring and Health Checks

### Built-in Endpoints

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (when enabled)
- **System Stats**: `GET /stats`

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600.5,
  "system_info": {
    "classifiers_available": ["pure_llm", "hybrid_rag"],
    "training_samples": 1005,
    "issue_types": 107,
    "categories": 8
  },
  "classifiers_loaded": {
    "pure_llm": true,
    "hybrid_rag": true
  }
}
```

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load test
locust -f load_test.py --host http://localhost:8000
```

## 🔒 Security Considerations

### API Keys

- Store API keys in environment variables
- Use different keys for development/production
- Rotate keys regularly

### Network Security

```bash
# Run behind reverse proxy (nginx)
server {
    listen 80;
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Rate Limiting

Configure in `production_config.yaml`:

```yaml
api:
  rate_limit:
    enabled: true
    requests_per_minute: 100
    burst: 20
```

## 📈 Performance Optimization

### Memory Management

```bash
# Monitor memory usage
docker stats ccms-api

# Adjust workers based on memory
# Rule of thumb: (RAM - 1GB) / 0.5GB per worker
```

### Caching

Redis caching is configured by default:

```yaml
cache:
  enabled: true
  type: "redis"
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
```

## 🐛 Troubleshooting

### Common Issues

**API won't start:**
```bash
# Check logs
docker logs ccms-api

# Verify data files exist
ls -la data/synthetic/combined_training_data.xlsx

# Test configuration
python -c "from classifier.config_manager import ConfigManager; ConfigManager().validate_config()"
```

**Out of memory:**
```bash
# Reduce workers
export WORKERS=2

# Monitor usage
htop
```

**Slow responses:**
```bash
# Check if embeddings are cached
ls -la data/embeddings/

# Monitor API performance
curl http://localhost:8000/stats
```

### Logs

```bash
# View API logs
tail -f logs/api.log

# Docker logs
docker-compose logs -f ccms-api

# System logs (if using systemd)
journalctl -u ccms-api -f
```

## 📦 API Endpoints

### Classification

```bash
# Single classification
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Request for extension of time due to weather delays",
    "approach": "hybrid_rag",
    "confidence_threshold": 0.7
  }'

# Batch classification
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["text1", "text2"],
    "approach": "hybrid_rag"
  }'
```

### System Information

```bash
# Get categories
curl http://localhost:8000/categories

# Get issue types
curl http://localhost:8000/issue-types

# Get system statistics
curl http://localhost:8000/stats
```

## 🔄 Updates and Maintenance

### Data Updates

```bash
# Update training data
cp new_training_data.xlsx data/synthetic/combined_training_data.xlsx

# Restart API to reload data
docker-compose restart ccms-api
```

### Code Updates

```bash
# Pull updates
git pull origin main

# Rebuild and restart
docker-compose up -d --build
```

## 📞 Support

- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health
- **Monitoring**: http://localhost:5555 (if Flower is enabled)

For issues, check logs and ensure all prerequisites are met.