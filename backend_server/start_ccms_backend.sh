#!/bin/bash

# ============================================================================
# CCMS Backend Startup Script for CCMS Classification System
# ============================================================================
# This script manages the startup of the CCMS backend server
# Features:
# - Detects running server instances
# - Warns user and optionally kills existing servers
# - Sets up environment variables
# - Starts fresh server instance
# - Provides status monitoring
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Paths relative to script location (backend_server/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"              # We are in the backend_server directory
VENV_DIR="$SCRIPT_DIR/../venv"         # Virtual env at project root level
ENV_FILE="$SCRIPT_DIR/.env"            # Environment file in backend_server/
PID_FILE="$BACKEND_DIR/server.pid"     # PID file in backend_server/
LOG_FILE="$BACKEND_DIR/server.log"     # Log file in backend_server/

# Default settings
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="5001"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if server is running
check_server_running() {
    local pids=$(ps aux | grep "python.*api/app.py" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        echo "$pids"
    else
        echo ""
    fi
}

# Function to check if port is in use
check_port_usage() {
    local port=${1:-$DEFAULT_PORT}
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:"$port" 2>/dev/null
    else
        netstat -tuln 2>/dev/null | grep ":$port " | wc -l
    fi
}

# Function to kill server processes
kill_servers() {
    local pids="$1"
    if [ -n "$pids" ]; then
        print_warning "Killing existing server processes: $pids"
        for pid in $pids; do
            kill -TERM "$pid" 2>/dev/null
            sleep 2
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "Force killing process $pid"
                kill -KILL "$pid" 2>/dev/null
            fi
        done
        sleep 3
    fi
}

# Function to kill processes using the port
kill_port_processes() {
    local port=${1:-$DEFAULT_PORT}
    print_status "Checking for processes using port $port..."
    
    if command -v lsof >/dev/null 2>&1; then
        local port_pids=$(lsof -ti:"$port" 2>/dev/null)
        if [ -n "$port_pids" ]; then
            print_warning "Killing processes using port $port: $port_pids"
            for pid in $port_pids; do
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    print_warning "Force killing process $pid using port $port"
                    kill -KILL "$pid" 2>/dev/null
                fi
            done
            sleep 2
        fi
    fi
}

# Function to clean up everything (processes and port)
cleanup_everything() {
    local port=${1:-$DEFAULT_PORT}
    print_status "Performing complete cleanup..."
    
    # Kill server processes
    local running_pids=$(check_server_running)
    if [ -n "$running_pids" ]; then
        kill_servers "$running_pids"
    fi
    
    # Kill any processes using the port
    kill_port_processes "$port"
    
    # Clean up PID file
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
        print_status "Removed old PID file"
    fi
    
    # Verify cleanup
    sleep 2
    local remaining_pids=$(check_server_running)
    local remaining_port=$(check_port_usage "$port")
    
    if [ -n "$remaining_pids" ]; then
        print_error "Some processes still running: $remaining_pids"
        return 1
    fi
    
    if [ -n "$remaining_port" ] && [ "$remaining_port" != "0" ]; then
        print_error "Port $port still in use"
        return 1
    fi
    
    print_success "Cleanup completed successfully"
    return 0
}

# Function to validate environment
validate_environment() {
    print_status "Validating environment..."
    
    # Check if directories exist
    if [ ! -d "$BACKEND_DIR" ]; then
        print_error "Backend directory not found: $BACKEND_DIR"
        exit 1
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found: $VENV_DIR"
        print_status "Please create virtual environment: python -m venv venv"
        exit 1
    fi
    
    if [ ! -f "$ENV_FILE" ]; then
        print_warning "Environment file not found: $ENV_FILE"
        print_status "Creating default environment file..."
        cat > "$ENV_FILE" << EOF
# CCMS Integrated Backend Environment Variables
FLASK_HOST=$DEFAULT_HOST
FLASK_PORT=$DEFAULT_PORT
FLASK_DEBUG=true
FLASK_USE_RELOADER=false

# API Keys (replace with your actual keys)
CLAUDE_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Backend URL for tests
BACKEND_URL=http://localhost:$DEFAULT_PORT
EOF
        print_warning "Please update $ENV_FILE with your actual API keys"
    fi
    
    # Check virtual environment activation
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        print_error "Virtual environment activation script not found"
        exit 1
    fi
}

# Function to load environment variables
load_environment() {
    print_status "Loading environment variables..."
    
    if [ -f "$ENV_FILE" ]; then
        # Export variables from .env file (excluding LOG_FILE to avoid conflicts)
        export $(grep -v '^#' "$ENV_FILE" | grep -v '^LOG_FILE=' | xargs)
        print_success "Environment variables loaded from $ENV_FILE"
    fi
    
    # Set defaults if not provided
    export FLASK_HOST=${FLASK_HOST:-$DEFAULT_HOST}
    export FLASK_PORT=${FLASK_PORT:-$DEFAULT_PORT}
    export FLASK_DEBUG=${FLASK_DEBUG:-true}
    
    # Derive BACKEND_URL from FLASK_HOST and FLASK_PORT
    # If FLASK_HOST is 0.0.0.0 (listen on all interfaces), use localhost for frontend connections
    local frontend_host="$FLASK_HOST"
    if [ "$FLASK_HOST" = "0.0.0.0" ]; then
        frontend_host="localhost"
    fi
    export BACKEND_URL="http://$frontend_host:$FLASK_PORT"
    
    # Validate required API keys
    local missing_keys=()
    [ -z "$CLAUDE_API_KEY" ] && missing_keys+=("CLAUDE_API_KEY")
    [ -z "$OPENAI_API_KEY" ] && missing_keys+=("OPENAI_API_KEY")
    [ -z "$GOOGLE_API_KEY" ] && missing_keys+=("GOOGLE_API_KEY")
    
    if [ ${#missing_keys[@]} -gt 0 ]; then
        print_warning "Missing API keys: ${missing_keys[*]}"
        print_status "The server will start but classification features may not work"
    fi
}

# Function to start the server
start_server() {
    print_status "Starting CCMS backend server..."
    
    # Activate virtual environment and start server
    cd "$BACKEND_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Clean old log file
    rm -f "$LOG_FILE"
    
    # Start server in background
    nohup python api/app.py > "$LOG_FILE" 2>&1 &
    local server_pid=$!
    
    # Save PID
    echo $server_pid > "$PID_FILE"
    
    print_status "Server starting with PID: $server_pid"
    print_status "Host: $FLASK_HOST"
    print_status "Port: $FLASK_PORT"
    print_status "Log file: $LOG_FILE"
    print_status "Backend URL: $BACKEND_URL"
    
    # Wait a moment and check if server started successfully
    sleep 5
    
    if kill -0 "$server_pid" 2>/dev/null; then
        print_success "Server process started successfully (PID: $server_pid)"
        
        # Verify port is acquired
        print_status "Verifying port $FLASK_PORT is acquired..."
        local port_check_attempts=0
        local max_attempts=10
        local port_acquired=false
        
        while [ $port_check_attempts -lt $max_attempts ]; do
            local port_usage=$(check_port_usage "$FLASK_PORT")
            if [ -n "$port_usage" ] && [ "$port_usage" != "0" ]; then
                port_acquired=true
                print_success "Port $FLASK_PORT successfully acquired!"
                break
            fi
            sleep 2
            port_check_attempts=$((port_check_attempts + 1))
            print_status "Waiting for port acquisition... ($port_check_attempts/$max_attempts)"
        done
        
        if [ "$port_acquired" = false ]; then
            print_error "Server started but failed to acquire port $FLASK_PORT"
            print_status "Check logs: tail -f $LOG_FILE"
            return 1
        fi
        
        # Final status report
        echo ""
        print_success "✅ SERVER STATUS REPORT ✅"
        echo "  🔧 Process ID: $server_pid"
        echo "  🌐 Host: $FLASK_HOST"
        echo "  🔌 Port: $FLASK_PORT (ACQUIRED)"
        echo "  📝 Log File: $LOG_FILE"
        echo "  🌍 Backend URL: $BACKEND_URL"
        
        # Test health endpoint
        print_status "Testing server health..."
        if command -v curl >/dev/null 2>&1; then
            sleep 5  # Give server time to fully initialize
            local health_response=$(curl -s "$BACKEND_URL/api/services/health" 2>/dev/null)
            if [ $? -eq 0 ] && echo "$health_response" | grep -q "healthy"; then
                print_success "🏥 Server health check passed!"
                print_status "You can now access the backend at: $BACKEND_URL"
                print_status "API documentation at: $BACKEND_URL/api"
            else
                print_warning "Server started but health check failed"
                print_status "Check logs: tail -f $LOG_FILE"
            fi
        else
            print_warning "curl not available, skipping health check"
            print_status "Server should be available at: $BACKEND_URL"
        fi
    else
        print_error "Server failed to start!"
        print_status "Check logs: cat $LOG_FILE"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -f, --force     Force kill existing servers without prompting"
    echo "  -k, --kill-only Kill existing servers and exit"
    echo "  -s, --status    Show server status and exit"
    echo "  --start         Start server (kill existing if found)"
    echo "  --stop          Stop server and clear port"
    echo "  --restart       Restart server (same as --start)"
    echo "  -t, --test      Run backend tests after starting server"
    echo ""
    echo "Environment Variables:"
    echo "  FLASK_HOST      Server host (default: $DEFAULT_HOST)"
    echo "  FLASK_PORT      Server port (default: $DEFAULT_PORT)"
    echo "  FLASK_DEBUG     Debug mode (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0              Start server (interactive mode)"
    echo "  $0 --start      Start server (auto-kill existing)"
    echo "  $0 --stop       Stop server and clear port"
    echo "  $0 --restart    Restart server (auto-kill existing)"
    echo "  $0 --force      Start server (kill existing without prompt)"
    echo "  $0 --status     Check server status"
    echo "  $0 --test       Start server and run tests"
}

# Function to get service status from API
get_service_status() {
    local backend_url="$1"
    if command -v curl >/dev/null 2>&1; then
        local health_response=$(curl -s "$backend_url/api/services/health" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "$health_response"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Function to show server status
show_status() {
    print_status "Checking server status..."
    
    local running_pids=$(check_server_running)
    local port_usage=$(check_port_usage "$FLASK_PORT")
    
    # Process Status
    echo ""
    echo "📊 Process Status:"
    if [ -n "$running_pids" ]; then
        print_warning "Server processes found: $running_pids"
        for pid in $running_pids; do
            local cmd=$(ps -p "$pid" -o command= 2>/dev/null)
            local start_time=$(ps -p "$pid" -o lstart= 2>/dev/null)
            echo "  PID $pid: Started $start_time"
            echo "    Command: $cmd"
        done
    else
        print_success "No server processes found"
    fi
    
    # Port Status
    echo ""
    echo "🔌 Port Status:"
    if [ -n "$port_usage" ] && [ "$port_usage" != "0" ]; then
        print_warning "Port $FLASK_PORT is in use"
    else
        print_success "Port $FLASK_PORT is available"
    fi
    
    # PID File Status
    echo ""
    echo "📄 PID File Status:"
    if [ -f "$PID_FILE" ]; then
        local saved_pid=$(cat "$PID_FILE")
        if kill -0 "$saved_pid" 2>/dev/null; then
            print_status "Saved PID file shows running process: $saved_pid"
        else
            print_warning "Saved PID file exists but process is not running"
            rm -f "$PID_FILE"
        fi
    else
        print_status "No PID file found"
    fi
    
    # Service Health Status (if server is running)
    if [ -n "$running_pids" ]; then
        echo ""
        echo "🏥 Service Health Status:"
        print_status "Checking service health at $BACKEND_URL..."
        
        local health_data=$(get_service_status "$BACKEND_URL")
        if [ -n "$health_data" ]; then
            # Parse health data and show services
            if command -v jq >/dev/null 2>&1; then
                # Use jq for nice formatting
                local overall_status=$(echo "$health_data" | jq -r '.status // "unknown"')
                echo "  Overall Status: $overall_status"
                echo ""
                echo "  🔧 Individual Services:"
                
                # Get service list
                local services=$(echo "$health_data" | jq -r '.services // {} | keys[]' 2>/dev/null)
                if [ -n "$services" ]; then
                    echo "$services" | while IFS= read -r service; do
                        local service_status=$(echo "$health_data" | jq -r ".services[\"$service\"].available // false")
                        local service_error=$(echo "$health_data" | jq -r ".services[\"$service\"].error // \"\"")
                        
                        if [ "$service_status" = "true" ]; then
                            echo "    ✅ $service: Available"
                        else
                            echo "    ❌ $service: Unavailable"
                            if [ -n "$service_error" ] && [ "$service_error" != "null" ] && [ "$service_error" != "" ]; then
                                echo "       Error: $service_error"
                            fi
                        fi
                    done
                else
                    print_warning "No service data found in health response"
                fi
            else
                # Fallback without jq - basic parsing
                print_status "Health response received (install jq for detailed parsing):"
                echo "$health_data" | head -10
            fi
            
            # Classification Service Detailed Status
            echo ""
            echo "  🤖 Classification Service Details:"
            local class_status=$(curl -s "$BACKEND_URL/api/services/hybrid-rag-classification/status" 2>/dev/null)
            if [ -n "$class_status" ] && command -v jq >/dev/null 2>&1; then
                local is_initialized=$(echo "$class_status" | jq -r '.data.is_initialized // false')
                local total_categories=$(echo "$class_status" | jq -r '.data.total_categories // 0')
                local total_issues=$(echo "$class_status" | jq -r '.data.total_issue_types // 0')
                local approaches=$(echo "$class_status" | jq -r '.data.available_approaches // [] | join(", ")')
                local service_name=$(echo "$class_status" | jq -r '.data.service_name // "Unknown"')
                
                echo "    Service: $service_name"
                echo "    Initialized: $is_initialized"
                echo "    Categories: $total_categories"
                echo "    Issue Types: $total_issues"
                echo "    Approaches: $approaches"
                
                if [ "$is_initialized" = "false" ]; then
                    local init_error=$(echo "$class_status" | jq -r '.data.initialization_error // ""')
                    if [ -n "$init_error" ] && [ "$init_error" != "null" ]; then
                        echo "    ⚠️  Initialization Error: $init_error"
                    fi
                fi
            else
                print_warning "Could not get classification service details"
            fi
            
            # Available Endpoints
            echo ""
            echo "  🌐 Available Endpoints:"
            echo ""
            echo "    📋 General:"
            echo "      Health Check: $BACKEND_URL/api/services/health"
            echo "      API Documentation: $BACKEND_URL/api"
            echo ""
            echo "    🤖 Classification Service:"
            echo "      Service Status: $BACKEND_URL/api/services/hybrid-rag-classification/status"
            echo "      Available Categories: $BACKEND_URL/api/services/hybrid-rag-classification/categories"
            echo "      Available Issue Types: $BACKEND_URL/api/services/hybrid-rag-classification/issue-types"
            echo "      Classify Text: $BACKEND_URL/api/services/hybrid-rag-classification/classify-text"
            echo "      Classify Batch: $BACKEND_URL/api/services/hybrid-rag-classification/classify-batch"
            echo "      Process Folder: $BACKEND_URL/api/services/hybrid-rag-classification/process-folder"
            echo "      Process Single PDF: $BACKEND_URL/api/services/hybrid-rag-classification/process-single-pdf"
            echo ""
            echo "    📋 Issue Management Services:"
            echo "      All Issue Types: $BACKEND_URL/api/services/issue-types"
            echo "      All Issue Categories: $BACKEND_URL/api/services/issue-categories"
            echo "      Categories by Issue Type: $BACKEND_URL/api/services/issue-categories/by-issue-type/{issue_type_id}"
            echo ""
            echo "    📄 Document Services:"
            echo "      Document Type: $BACKEND_URL/api/services/document-type/classify"
            echo "      OCR Extract: $BACKEND_URL/api/services/ocr/extract-text"
            echo "      OCR Methods: $BACKEND_URL/api/services/ocr/methods"
            echo "      Extract Pages: $BACKEND_URL/api/services/ocr/extract-text-from-pages"
            echo ""
            echo "    🧠 LLM Services:"
            echo "      Extract Structured: $BACKEND_URL/api/services/llm/extract-structured"
            echo "      Generate Text: $BACKEND_URL/api/services/llm/generate-text"
            echo "      Summarize: $BACKEND_URL/api/services/llm/summarize"
            echo ""
            echo "    🗂️ Category Mapping:"
            echo "      Get Mappings: $BACKEND_URL/api/services/category-mapping/mappings"
            echo "      Map Issue: $BACKEND_URL/api/services/category-mapping/map-issue"
            echo "      Map Categories: $BACKEND_URL/api/services/category-mapping/map-categories"
            echo "      Categories by Issue: $BACKEND_URL/api/services/category-mapping/categories/{category}/issues"
            
            # Quick endpoint availability test
            echo ""
            echo "  🔍 Endpoint Availability Test:"
            if command -v curl >/dev/null 2>&1; then
                # Test key endpoints
                local test_endpoints=(
                    "/api/services/health:Health"
                    "/api/services/hybrid-rag-classification/status:Classification Status"
                    "/api/services/hybrid-rag-classification/categories:Categories"
                    "/api/services/hybrid-rag-classification/issue-types:Issue Types"
                    "/api/services/issue-types:All Issue Types"
                    "/api/services/issue-categories:All Issue Categories"
                    "/api/services/ocr/methods:OCR Methods"
                )
                
                for endpoint_info in "${test_endpoints[@]}"; do
                    local endpoint="${endpoint_info%%:*}"
                    local name="${endpoint_info##*:}"
                    local response=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL$endpoint" 2>/dev/null)
                    
                    if [ "$response" = "200" ]; then
                        echo "    ✅ $name ($endpoint)"
                    elif [ "$response" = "404" ]; then
                        echo "    ❌ $name ($endpoint) - Not Found"
                    elif [ "$response" = "500" ]; then
                        echo "    ⚠️  $name ($endpoint) - Server Error"
                    else
                        echo "    🔄 $name ($endpoint) - Response: $response"
                    fi
                done
            else
                echo "    ⚠️  curl not available - cannot test endpoints"
            fi
            
        else
            print_error "Could not connect to server health endpoint"
            print_status "Server may be starting up or experiencing issues"
        fi
    else
        echo ""
        print_status "Server not running - cannot check service health"
    fi
    
    # Environment Status
    echo ""
    echo "🌍 Environment Status:"
    echo "  Host: ${FLASK_HOST:-$DEFAULT_HOST}"
    echo "  Port: ${FLASK_PORT:-$DEFAULT_PORT}"
    echo "  Debug: ${FLASK_DEBUG:-true}"
    echo ""
    echo "🔗 Frontend Integration:"
    echo "  ✅ Backend URL: $BACKEND_URL"
    echo "     Use this URL in your frontend application to connect to the backend"
    echo "     Example: fetch('$BACKEND_URL/api/services/health')"
    echo ""
    echo "📝 Logs:"
    echo "  Log File: $LOG_FILE"
    if [ -f "$LOG_FILE" ]; then
        local log_size=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
        echo "  Log Lines: $log_size"
        echo "  Last Modified: $(stat -f "%Sm" "$LOG_FILE" 2>/dev/null || echo "Unknown")"
    else
        echo "  Log Status: No log file found"
    fi
    
    # API Key Status
    echo ""
    echo "🔑 API Key Status:"
    [ -n "$CLAUDE_API_KEY" ] && echo "  ✅ CLAUDE_API_KEY: Set" || echo "  ❌ CLAUDE_API_KEY: Missing"
    [ -n "$OPENAI_API_KEY" ] && echo "  ✅ OPENAI_API_KEY: Set" || echo "  ❌ OPENAI_API_KEY: Missing"
    [ -n "$GOOGLE_API_KEY" ] && echo "  ✅ GOOGLE_API_KEY: Set" || echo "  ❌ GOOGLE_API_KEY: Missing"
}

# Main script
main() {
    echo "============================================================================"
    echo "CCMS Integrated Backend Startup Script"
    echo "============================================================================"
    
    # Parse command line arguments
    local force_kill=false
    local kill_only=false
    local status_only=false
    local stop_only=false
    local run_tests=false
    local auto_start=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -f|--force)
                force_kill=true
                shift
                ;;
            -k|--kill-only)
                kill_only=true
                shift
                ;;
            -s|--status)
                status_only=true
                shift
                ;;
            --stop)
                stop_only=true
                shift
                ;;
            --start|--restart)
                auto_start=true
                shift
                ;;
            -t|--test)
                run_tests=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate environment first
    validate_environment
    load_environment
    
    # Show status if requested
    if [ "$status_only" = true ]; then
        show_status
        exit 0
    fi
    
    # Stop server if requested
    if [ "$stop_only" = true ]; then
        print_status "Stopping CCMS backend server..."
        cleanup_everything "$FLASK_PORT"
        if [ $? -eq 0 ]; then
            print_success "Server stopped and port cleared successfully!"
        else
            print_error "Failed to stop server or clear port"
            exit 1
        fi
        exit 0
    fi
    
    # Handle server cleanup based on options
    if [ "$auto_start" = true ]; then
        # Auto start mode - always clean up everything
        print_status "Auto-start mode: cleaning up existing servers and ports..."
        cleanup_everything "$FLASK_PORT"
        if [ $? -ne 0 ]; then
            print_error "Failed to clean up existing servers/ports"
            exit 1
        fi
    else
        # Interactive/force mode - check for running servers
        local running_pids=$(check_server_running)
        
        if [ -n "$running_pids" ]; then
            print_warning "Found existing server processes: $running_pids"
            
            if [ "$force_kill" = true ]; then
                cleanup_everything "$FLASK_PORT"
            else
                echo -n "Do you want to kill existing servers? (y/N): "
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    cleanup_everything "$FLASK_PORT"
                else
                    print_status "Exiting without starting new server"
                    exit 0
                fi
            fi
        fi
    fi
    
    # If kill-only mode, exit after killing
    if [ "$kill_only" = true ]; then
        print_success "Servers killed. Exiting."
        exit 0
    fi
    
    # Start the server
    if ! start_server; then
        print_error "Failed to start server successfully!"
        echo "============================================================================"
        exit 1
    fi
    
    # Show status after successful startup when using --start
    if [ "$auto_start" = true ]; then
        echo ""
        echo "============================================================================"
        print_success "Server started successfully! Here's the current status:"
        echo "============================================================================"
        show_status
    fi
    
    # Run tests if requested
    if [ "$run_tests" = true ]; then
        print_status "Running backend tests..."
        sleep 5  # Give server time to fully start
        cd "$SCRIPT_DIR"
        source "$VENV_DIR/bin/activate"
        python test_integrated_backend.py
    fi
    
    print_success "Backend startup completed!"
    echo "============================================================================"
}

# Run main function
main "$@"