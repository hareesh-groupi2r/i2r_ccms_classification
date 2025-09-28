#!/bin/bash

# =============================================================================
# Batch Processing Script for Lots 21-27
# Processes each lot individually with appropriate ground truth files
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOTS_BASE_DIR="$SCRIPT_DIR/data/Lots21-27"
RESULTS_BASE_DIR="$SCRIPT_DIR/results"
INTEGRATED_BACKEND_URL="http://localhost:5001"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-$NC}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() { log "ERROR: $1" "$RED" >&2; }
success() { log "SUCCESS: $1" "$GREEN"; }
warn() { log "WARNING: $1" "$YELLOW"; }
info() { log "INFO: $1" "$BLUE"; }

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check if lots directory exists
    if [[ ! -d "$LOTS_BASE_DIR" ]]; then
        error "Lots directory not found at $LOTS_BASE_DIR"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_BASE_DIR"
    
    # Check if integrated backend is running
    if ! curl -s "$INTEGRATED_BACKEND_URL/api/services/health" > /dev/null 2>&1; then
        error "Integrated backend not running at $INTEGRATED_BACKEND_URL"
        error "Please start the integrated backend server first:"
        error "  cd integrated_backend && python api/app.py"
        exit 1
    else
        info "✅ Integrated backend is running"
    fi
    
    success "Prerequisites checked"
}

# API call to integrated backend
call_integrated_backend() {
    local pdf_folder="$1"
    local ground_truth_file="$2" 
    local output_folder="$3"
    local enable_metrics="$4"
    
    # Build JSON payload
    local payload
    payload=$(cat << EOF
{
    "pdf_folder": "$pdf_folder",
    "output_folder": "$output_folder",
    "options": {
        "approaches": ["hybrid_rag"],
        "confidence_threshold": 0.3,
        "max_pages": 2
    },
    "enable_metrics": $enable_metrics
EOF
    )
    
    # Add ground truth if provided
    if [[ -n "$ground_truth_file" && -f "$ground_truth_file" ]]; then
        payload=$(echo "$payload" | sed 's/$/,/' | sed '$s/,$//')
        payload="$payload,\n    \"ground_truth_file\": \"$ground_truth_file\"\n}"
    else
        payload="$payload}"
    fi
    
    info "Calling integrated backend API..."
    info "Payload: $(echo -e "$payload" | jq -c '.' 2>/dev/null || echo -e "$payload")"
    
    # Make API call
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$(echo -e "$payload")" \
        "$INTEGRATED_BACKEND_URL/api/services/hybrid-rag-classification/process-folder")
    
    local curl_exit_code=$?
    
    if [[ $curl_exit_code -ne 0 ]]; then
        error "API call failed with curl exit code: $curl_exit_code"
        return 1
    fi
    
    # Check if response indicates success
    if echo "$response" | jq -e '.processing_stats' > /dev/null 2>&1; then
        info "✅ API call successful"
        return 0
    else
        error "API call failed. Response: $response"
        return 1
    fi
}

# Process individual lot
process_lot() {
    local lot_name="$1"
    local pdf_folder="$2"
    local ground_truth_file="$3"
    local output_folder="$4"
    local enable_llm="${5:-false}" # Currently unused since integrated backend uses hybrid_rag only
    
    info "Processing $lot_name..."
    
    # Check if PDF folder exists
    if [[ ! -d "$pdf_folder" ]]; then
        error "$lot_name: PDF folder not found at $pdf_folder"
        return 1
    fi
    
    # Count PDF files
    local pdf_count
    pdf_count=$(find "$pdf_folder" -name "*.pdf" | wc -l)
    info "$lot_name: Found $pdf_count PDF files"
    
    if [[ $pdf_count -eq 0 ]]; then
        warn "$lot_name: No PDF files found, skipping..."
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_folder"
    
    # Prepare parameters
    local enable_metrics="true"
    if [[ -n "$ground_truth_file" && -f "$ground_truth_file" ]]; then
        info "$lot_name: Using ground truth file: $ground_truth_file"
    else
        info "$lot_name: No ground truth file, metrics will be limited"
    fi
    
    info "$lot_name: Using Hybrid RAG approach via integrated backend"
    
    # Call integrated backend API
    if call_integrated_backend "$pdf_folder" "$ground_truth_file" "$output_folder" "$enable_metrics"; then
        success "$lot_name: Processing completed successfully"
        return 0
    else
        error "$lot_name: Processing failed"
        return 1
    fi
}

# Main processing function
main() {
    info "🚀 Starting Lots 21-27 Batch Processing"
    info "="*60
    
    check_prerequisites
    
    local total_lots=0
    local successful_lots=0
    local failed_lots=0
    
    # Process Lots 21-23
    info "Processing Lots 21-23..."
    
    # LOT-21 (has ground truth)
    if [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-21" ]]; then
        ((total_lots++))
        local lot21_gt="$LOTS_BASE_DIR/Lot 21 to 23/LOT-21/LOT-21.xlsx"
        if process_lot "LOT-21" \
                      "$LOTS_BASE_DIR/Lot 21 to 23/LOT-21" \
                      "$lot21_gt" \
                      "$RESULTS_BASE_DIR/LOT-21" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-21 directory not found"
    fi
    
    # LOT-22 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-22" ]]; then
        ((total_lots++))
        if process_lot "LOT-22" \
                      "$LOTS_BASE_DIR/Lot 21 to 23/LOT-22" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-22" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-22 directory not found"
    fi
    
    # LOT-23 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-23" ]]; then
        ((total_lots++))
        if process_lot "LOT-23" \
                      "$LOTS_BASE_DIR/Lot 21 to 23/LOT-23" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-23" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-23 directory not found"
    fi
    
    # Process Lots 24-27
    info "Processing Lots 24-27..."
    
    # LOT-24 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 24" ]]; then
        ((total_lots++))
        if process_lot "LOT-24" \
                      "$LOTS_BASE_DIR/Lot 24 to 27/Lot 24" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-24" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-24 directory not found"
    fi
    
    # LOT-25 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 25" ]]; then
        ((total_lots++))
        if process_lot "LOT-25" \
                      "$LOTS_BASE_DIR/Lot 24 to 27/Lot 25" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-25" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-25 directory not found"
    fi
    
    # LOT-26 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 26" ]]; then
        ((total_lots++))
        if process_lot "LOT-26" \
                      "$LOTS_BASE_DIR/Lot 24 to 27/Lot 26" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-26" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-26 directory not found"
    fi
    
    # LOT-27 (no ground truth found)
    if [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 27" ]]; then
        ((total_lots++))
        if process_lot "LOT-27" \
                      "$LOTS_BASE_DIR/Lot 24 to 27/Lot 27" \
                      "" \
                      "$RESULTS_BASE_DIR/LOT-27" \
                      "false"; then
            ((successful_lots++))
        else
            ((failed_lots++))
        fi
    else
        warn "LOT-27 directory not found"
    fi
    
    # Final summary
    echo ""
    info "="*60
    info "📊 BATCH PROCESSING SUMMARY"
    info "="*60
    info "Total Lots: $total_lots"
    success "Successful: $successful_lots"
    if [[ $failed_lots -gt 0 ]]; then
        error "Failed: $failed_lots"
    else
        info "Failed: $failed_lots"
    fi
    
    info "📁 Results saved to: $RESULTS_BASE_DIR"
    info "📄 Individual lot results available in respective subdirectories"
    
    if [[ $failed_lots -gt 0 ]]; then
        error "Some lots failed processing. Check logs above for details."
        return 1
    else
        success "All lots processed successfully! 🎉"
        return 0
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Batch Processing Script for Lots 21-27"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h        Show this help message"
        echo "  --with-llm        Enable Pure LLM approach for all lots"
        echo "  --dry-run         Show what would be processed without running"
        echo ""
        echo "The script will automatically:"
        echo "  - Process all lots (21-27) with appropriate PDF folders"
        echo "  - Use ground truth files where available (LOT-21.xlsx)"
        echo "  - Save results to individual folders under results/"
        echo "  - Use Hybrid RAG approach by default (faster, reliable)"
        echo ""
        exit 0
        ;;
    --with-llm)
        # Enable LLM for all lots (modify the script to set enable_llm=true)
        warn "Pure LLM approach will be enabled for all lots (slower but potentially more comprehensive)"
        # This would require modifying the process_lot calls above
        ;;
    --dry-run)
        info "DRY RUN: Would process the following lots:"
        [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-21" ]] && echo "  - LOT-21 (with ground truth: LOT-21.xlsx)"
        [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-22" ]] && echo "  - LOT-22 (no ground truth)"
        [[ -d "$LOTS_BASE_DIR/Lot 21 to 23/LOT-23" ]] && echo "  - LOT-23 (no ground truth)"
        [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 24" ]] && echo "  - LOT-24 (no ground truth)"
        [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 25" ]] && echo "  - LOT-25 (no ground truth)"
        [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 26" ]] && echo "  - LOT-26 (no ground truth)"
        [[ -d "$LOTS_BASE_DIR/Lot 24 to 27/Lot 27" ]] && echo "  - LOT-27 (no ground truth)"
        info "Results would be saved to: $RESULTS_BASE_DIR"
        exit 0
        ;;
    "")
        # Run normally
        main
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac