# CCMS Classification Service Management

This document provides instructions for managing the CCMS Classification API backend service using the provided service management script.

## Overview

The service management system provides a comprehensive solution for starting, stopping, monitoring, and troubleshooting the CCMS Classification API backend service. It includes:

- **service.sh**: Main service management script
- **Docker support**: Container-based deployment
- **Systemd integration**: Linux service management
- **Health monitoring**: Automated health checks
- **Log management**: Centralized logging and monitoring

## Quick Start

### 1. Basic Service Operations

```bash
# Start the service
./service.sh start

# Check service status
./service.sh status

# Stop the service
./service.sh stop

# Restart the service
./service.sh restart
```

### 2. Log Management

```bash
# View last 50 lines of logs (default)
./service.sh logs

# View last 100 lines of logs
./service.sh logs 100

# Follow logs in real-time
./service.sh follow
```

### 3. Health Monitoring

```bash
# Test API endpoints
./service.sh test

# Check detailed status with health check
./service.sh status
```

## Configuration

### Environment Variables

The service can be configured using environment variables:

```bash
# Set environment mode (development/production)
export ENVIRONMENT=production

# Set server port
export PORT=8000

# Set server host
export HOST=0.0.0.0

# Number of worker processes (production only)
export WORKERS=4
```

### API Keys Configuration

Create or update the `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GEMINI_API_KEY=your-gemini-key-here

# Server Configuration
ENVIRONMENT=development
PORT=8000
HOST=127.0.0.1
WORKERS=4

# Logging
LOG_LEVEL=INFO
```

## Deployment Options

### 1. Development Mode

For development and testing:

```bash
# Start in development mode with auto-reload
./service.sh start

# The service will start on http://127.0.0.1:8000
# API documentation available at http://127.0.0.1:8000/docs
```

### 2. Production Mode

For production deployment:

```bash
# Start in production mode
ENVIRONMENT=production ./service.sh start

# Or set environment variable permanently
export ENVIRONMENT=production
./service.sh start
```

### 3. Docker Deployment

Using Docker Compose:

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f ccms-api

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### 4. Systemd Service (Linux)

For Linux systems with systemd:

```bash
# Copy service file to systemd directory
sudo cp ccms-classification-api.service /etc/systemd/system/

# Update paths in the service file to match your installation
sudo nano /etc/systemd/system/ccms-classification-api.service

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable ccms-classification-api.service

# Start service
sudo systemctl start ccms-classification-api.service

# Check status
sudo systemctl status ccms-classification-api.service
```

## Service Management Features

### Health Checks

The service includes comprehensive health checks:

- **Startup Health Check**: Verifies service starts correctly
- **API Endpoint Testing**: Tests critical endpoints
- **Process Monitoring**: Monitors process status
- **Automatic Recovery**: Restarts on failure (systemd/docker)

### Log Management

Centralized logging with rotation:

- **Service Management Logs**: `logs/service.log`
- **API Application Logs**: `logs/api.log`
- **Automatic Rotation**: Prevents log files from growing too large

### Process Management

- **PID File Management**: Tracks running processes
- **Graceful Shutdown**: Handles SIGTERM signals properly
- **Process Cleanup**: Removes stale PID files
- **Resource Monitoring**: Shows process resource usage

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check for errors in logs
./service.sh logs

# Verify virtual environment
ls -la venv/

# Check if port is already in use
netstat -tlnp | grep :8000
# or
ss -tlnp | grep :8000

# Check API keys
cat .env | grep API_KEY
```

#### 2. Service Stops Unexpectedly

```bash
# Check recent logs for errors
./service.sh logs 100

# Monitor system resources
top
free -h
df -h

# Check for Python errors
./service.sh follow
```

#### 3. Health Check Failures

```bash
# Test individual endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/categories

# Check service status
./service.sh status

# Restart service
./service.sh restart
```

### Log Analysis

Important log patterns to look for:

- **Startup Issues**: Look for "ERROR" during startup
- **API Key Problems**: Check for authentication errors
- **Model Loading**: Monitor "Loading" and "initialized" messages
- **Request Errors**: Monitor HTTP error codes

### Performance Monitoring

Monitor service performance:

```bash
# Check process resources
./service.sh status

# Monitor API response times
curl -w "@curl-format.txt" http://localhost:8000/health

# Check system resources
htop
iotop
```

## API Endpoints

Once the service is running, these endpoints are available:

### Core Endpoints

- **Health Check**: `GET /health`
- **API Documentation**: `GET /docs`
- **Classification**: `POST /classify`
- **Batch Classification**: `POST /classify/batch`

### Information Endpoints

- **Categories**: `GET /categories`
- **Issue Types**: `GET /issue-types`
- **System Statistics**: `GET /stats`

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Get available categories
curl http://localhost:8000/categories

# Classify text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We need extension of time for the project completion due to unexpected delays",
    "approach": "hybrid_rag",
    "confidence_threshold": 0.7
  }'
```

## Security Considerations

### Production Security

- **API Keys**: Store in environment variables, not code
- **Network Access**: Restrict to necessary ports
- **User Permissions**: Run service with minimal privileges
- **Log Sanitization**: Avoid logging sensitive information
- **HTTPS**: Use reverse proxy with SSL/TLS

### Container Security

```bash
# Build with security updates
docker-compose build --no-cache

# Run security scan
docker scan ccms-classification-api

# Update base images regularly
docker-compose pull
docker-compose up -d
```

## Monitoring and Maintenance

### Regular Maintenance Tasks

1. **Log Rotation**: Monitor log file sizes
2. **Dependency Updates**: Keep Python packages updated
3. **Model Updates**: Update training data and models
4. **Performance Monitoring**: Monitor resource usage
5. **Security Updates**: Apply system security patches

### Monitoring Setup

For production environments, consider:

- **System Monitoring**: Prometheus + Grafana
- **Log Aggregation**: ELK Stack or similar
- **Uptime Monitoring**: External monitoring services
- **Alert Management**: Automated alerts for failures

## Support and Troubleshooting

If you encounter issues:

1. **Check Logs**: Start with `./service.sh logs`
2. **Verify Configuration**: Ensure API keys and settings are correct
3. **Test Components**: Use `./service.sh test` to verify functionality
4. **Resource Check**: Monitor CPU, memory, and disk usage
5. **Network Check**: Verify ports are accessible

For additional support, check the project documentation and logs for specific error messages.