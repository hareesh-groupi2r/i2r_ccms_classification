#!/bin/bash

# =============================================================================
# CCMS Classification Service Management Script
# Provides start, stop, restart, status, and log management for the backend API
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SERVICE_NAME="ccms-classification-api"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PID_FILE="$PROJECT_DIR/logs/${SERVICE_NAME}.pid"
LOG_FILE="$PROJECT_DIR/logs/service.log"
API_LOG_FILE="$PROJECT_DIR/logs/api.log"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-$NC}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Error logging
error() {
    log "ERROR: $1" "$RED" >&2
}

# Success logging
success() {
    log "SUCCESS: $1" "$GREEN"
}

# Warning logging
warn() {
    log "WARNING: $1" "$YELLOW"
}

# Info logging
info() {
    log "INFO: $1" "$BLUE"
}

# Check if service is running
is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is not running
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Get process ID if running
get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# Setup environment
setup_environment() {
    info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/data/embeddings"
    mkdir -p "$PROJECT_DIR/data/models"
    mkdir -p "$PROJECT_DIR/data/backups"
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found at $VENV_DIR"
        error "Please create it with: python -m venv venv"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
        warn ".env file not found. Creating template..."
        cat > "$PROJECT_DIR/.env" << EOF
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
EOF
        warn "Please edit .env file with your API keys before starting the service"
    fi
    
    success "Environment setup complete"
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    info "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            success "Service is healthy and responding"
            return 0
        fi
        
        if [[ $attempt -eq 1 ]]; then
            info "Waiting for service to start..."
        fi
        
        sleep 2
        ((attempt++))
    done
    
    error "Health check failed - service not responding after $max_attempts attempts"
    return 1
}

# Start the service
start_service() {
    info "Starting $SERVICE_NAME..."
    
    if is_running; then
        warn "Service is already running (PID: $(get_pid))"
        return 0
    fi
    
    # Setup environment
    setup_environment
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Activate virtual environment and start service
    info "Activating virtual environment and starting service..."
    
    # Start service in background
    (
        source "$VENV_DIR/bin/activate"
        
        # Set environment variables
        export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
        export ENVIRONMENT="$ENVIRONMENT"
        export PORT="$PORT"
        export HOST="$HOST"
        
        # Start the service
        if [[ "$ENVIRONMENT" == "production" ]]; then
            # Production mode with proper startup script
            python start_production.py >> "$API_LOG_FILE" 2>&1 &
        else
            # Development mode with direct uvicorn
            python -m uvicorn production_api:app \
                --host "$HOST" \
                --port "$PORT" \
                --log-level info \
                --access-log >> "$API_LOG_FILE" 2>&1 &
        fi
        
        echo $! > "$PID_FILE"
    ) &
    
    # Wait a moment for the service to start
    sleep 3
    
    # Check if service started successfully
    if is_running; then
        success "Service started successfully (PID: $(get_pid))"
        info "API Documentation available at: http://$HOST:$PORT/docs"
        info "Service logs: tail -f $API_LOG_FILE"
        
        # Perform health check
        if health_check; then
            success "Service is ready to accept requests"
        else
            error "Service started but health check failed"
            stop_service
            exit 1
        fi
    else
        error "Failed to start service"
        error "Check logs: tail -n 50 $API_LOG_FILE"
        exit 1
    fi
}

# Stop the service
stop_service() {
    info "Stopping $SERVICE_NAME..."
    
    if ! is_running; then
        warn "Service is not running"
        return 0
    fi
    
    local pid
    pid=$(get_pid)
    
    info "Sending SIGTERM to process $pid..."
    kill "$pid" 2>/dev/null || {
        warn "Process $pid not found or already terminated"
        rm -f "$PID_FILE"
        return 0
    }
    
    # Wait for graceful shutdown
    local attempts=30
    while [[ $attempts -gt 0 ]] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        ((attempts--))
    done
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        warn "Graceful shutdown failed, force killing process $pid..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 2
    fi
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    success "Service stopped successfully"
}

# Restart the service
restart_service() {
    info "Restarting $SERVICE_NAME..."
    stop_service
    sleep 2
    start_service
}

# Show service status
show_status() {
    info "Checking $SERVICE_NAME status..."
    
    if is_running; then
        local pid
        pid=$(get_pid)
        success "Service is running (PID: $pid)"
        
        # Show process information
        if command -v ps >/dev/null 2>&1; then
            echo ""
            echo "Process Information:"
            ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || true
        fi
        
        # Show network information
        if command -v netstat >/dev/null 2>&1; then
            echo ""
            echo "Network Information:"
            netstat -tlnp 2>/dev/null | grep ":$PORT " || echo "Port $PORT not found in netstat output"
        elif command -v ss >/dev/null 2>&1; then
            echo ""
            echo "Network Information:"
            ss -tlnp | grep ":$PORT " || echo "Port $PORT not found in ss output"
        fi
        
        # Perform health check
        echo ""
        if health_check; then
            success "Service health check passed"
        else
            error "Service health check failed"
        fi
        
    else
        error "Service is not running"
        
        # Check for recent logs
        if [[ -f "$API_LOG_FILE" ]]; then
            echo ""
            echo "Recent log entries (last 10 lines):"
            tail -n 10 "$API_LOG_FILE" 2>/dev/null || echo "No recent logs found"
        fi
    fi
}

# Show service logs
show_logs() {
    local lines="${1:-50}"
    
    info "Showing last $lines lines of service logs..."
    
    if [[ -f "$API_LOG_FILE" ]]; then
        echo ""
        echo "=== API Logs ($API_LOG_FILE) ==="
        tail -n "$lines" "$API_LOG_FILE"
    else
        warn "API log file not found: $API_LOG_FILE"
    fi
    
    if [[ -f "$LOG_FILE" ]]; then
        echo ""
        echo "=== Service Management Logs ($LOG_FILE) ==="
        tail -n "$lines" "$LOG_FILE"
    else
        warn "Service log file not found: $LOG_FILE"
    fi
}

# Follow service logs
follow_logs() {
    info "Following service logs (Ctrl+C to stop)..."
    
    if [[ -f "$API_LOG_FILE" ]]; then
        tail -f "$API_LOG_FILE"
    else
        error "API log file not found: $API_LOG_FILE"
        exit 1
    fi
}

# Test API endpoints
test_api() {
    info "Testing API endpoints..."
    
    if ! is_running; then
        error "Service is not running. Start it first with: $0 start"
        exit 1
    fi
    
    local base_url="http://$HOST:$PORT"
    
    echo ""
    echo "Testing endpoints:"
    
    # Test health endpoint
    echo -n "  Health check: "
    if curl -s -f "$base_url/health" > /dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    
    # Test categories endpoint
    echo -n "  Categories: "
    if curl -s -f "$base_url/categories" > /dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    
    # Test issue types endpoint
    echo -n "  Issue types: "
    if curl -s -f "$base_url/issue-types" > /dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    
    # Test stats endpoint
    echo -n "  Statistics: "
    if curl -s -f "$base_url/stats" > /dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "${RED}✗ FAILED${NC}"
    fi
    
    echo ""
    success "API endpoint testing completed"
}

# Show help
show_help() {
    echo "CCMS Classification Service Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start          Start the service"
    echo "  stop           Stop the service"
    echo "  restart        Restart the service"
    echo "  status         Show service status"
    echo "  logs [LINES]   Show service logs (default: 50 lines)"
    echo "  follow         Follow service logs in real-time"
    echo "  test           Test API endpoints"
    echo "  help           Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT    Set to 'production' for production mode (default: development)"
    echo "  PORT          Server port (default: 8000)"
    echo "  HOST          Server host (default: 127.0.0.1)"
    echo ""
    echo "Examples:"
    echo "  $0 start                 # Start the service"
    echo "  $0 status                # Check service status"
    echo "  $0 logs 100             # Show last 100 log lines"
    echo "  ENVIRONMENT=production $0 start  # Start in production mode"
    echo ""
    echo "Log Files:"
    echo "  Service Management: $LOG_FILE"
    echo "  API Logs:          $API_LOG_FILE"
    echo ""
}

# Main script logic
main() {
    # Ensure logs directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    case "${1:-help}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-50}"
            ;;
        follow)
            follow_logs
            ;;
        test)
            test_api
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"