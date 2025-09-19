#!/bin/bash

# =============================================================================
# Single Lot Processing Script for Lots 21-27
# Process individual lots with custom options
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
LOTS_BASE_DIR="$SCRIPT_DIR/data/Lots21-27"
RESULTS_BASE_DIR="$SCRIPT_DIR/results"
INTEGRATED_BACKEND_URL="http://localhost:5001"
BACKEND_SCRIPT="$SCRIPT_DIR/start_integrated_backend.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${2:-$NC}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { log "ERROR: $1" "$RED" >&2; }
success() { log "SUCCESS: $1" "$GREEN"; }
warn() { log "WARNING: $1" "$YELLOW"; }
info() { log "INFO: $1" "$BLUE"; }

show_help() {
    cat << EOF
Single Lot Processing Script for Lots 21-27

Usage: $0 LOT_NUMBER [OPTIONS]

Arguments:
  LOT_NUMBER        The lot number to process (21, 22, 23, 24, 25, 26, or 27)

Options:
  --with-llm        Enable Pure LLM approach (slower but comprehensive)
  --no-metrics      Disable metrics calculation
  --output DIR      Custom output directory (default: results/LOT-XX)
  --limit N         Process only first N files (default: all files)
  --help, -h        Show this help message

Examples:
  $0 21                    # Process LOT-21 with default settings
  $0 24 --with-llm        # Process LOT-24 with Pure LLM enabled
  $0 22 --no-metrics      # Process LOT-22 without metrics
  $0 25 --output my_results/lot25  # Custom output directory
  $0 21 --limit 5         # Process only first 5 files from LOT-21

Available Lots:
EOF
    
    # List available lots
    for lot in 21 22 23 24 25 26 27; do
        local lot_dir=""
        if [[ $lot -le 23 ]]; then
            lot_dir="$LOTS_BASE_DIR/Lot 21 to 23/LOT-$lot"
        else
            lot_dir="$LOTS_BASE_DIR/Lot 24 to 27/Lot $lot"
        fi
        
        if [[ -d "$lot_dir" ]]; then
            local pdf_count
            pdf_count=$(find "$lot_dir" -name "*.pdf" | wc -l)
            local gt_info="no ground truth"
            
            if [[ $lot -eq 21 ]] && [[ -f "$lot_dir/LOT-21.xlsx" ]]; then
                gt_info="with ground truth (LOT-21.xlsx)"
            fi
            
            echo "  LOT-$lot: $pdf_count PDFs ($gt_info)"
        else
            echo "  LOT-$lot: directory not found"
        fi
    done
}

get_lot_info() {
    local lot_num="$1"
    local lot_dir=""
    local ground_truth=""
    
    # Determine lot directory based on lot number
    if [[ $lot_num -ge 21 && $lot_num -le 23 ]]; then
        lot_dir="$LOTS_BASE_DIR/Lot 21 to 23/LOT-$lot_num"
    elif [[ $lot_num -ge 24 && $lot_num -le 27 ]]; then
        lot_dir="$LOTS_BASE_DIR/Lot 24 to 27/Lot $lot_num"
    else
        error "Invalid lot number: $lot_num. Must be 21-27." >&2
        exit 1
    fi
    
    if [[ ! -d "$lot_dir" ]]; then
        error "Lot directory not found: $lot_dir" >&2
        exit 1
    fi
    
    # Auto-detect ground truth Excel file in the lot directory
    # Use null delimiter to handle filenames with spaces
    local excel_count
    excel_count=$(find "$lot_dir" -maxdepth 1 \( -name "*.xlsx" -o -name "*.xls" \) -print0 2>/dev/null | grep -c .)
    
    if [[ $excel_count -eq 1 ]]; then
        # Exactly one Excel file found - use it as ground truth
        ground_truth=$(find "$lot_dir" -maxdepth 1 \( -name "*.xlsx" -o -name "*.xls" \) -print 2>/dev/null)
        info "📊 Auto-detected ground truth file: $(basename "$ground_truth")" >&2
    elif [[ $excel_count -gt 1 ]]; then
        # Multiple Excel files - apply pattern matching
        local patterns=("EDMS*.xlsx" "LOT-*.xlsx" "*ground*truth*.xlsx" "*labels*.xlsx")
        for pattern in "${patterns[@]}"; do
            local match_count
            match_count=$(find "$lot_dir" -maxdepth 1 -name "$pattern" -print0 2>/dev/null | grep -c .)
            if [[ $match_count -eq 1 ]]; then
                ground_truth=$(find "$lot_dir" -maxdepth 1 -name "$pattern" -print 2>/dev/null)
                info "📊 Pattern-matched ground truth file: $(basename "$ground_truth")" >&2
                break
            fi
        done
        
        # If no pattern match, use the first Excel file alphabetically
        if [[ -z "$ground_truth" ]]; then
            ground_truth=$(find "$lot_dir" -maxdepth 1 \( -name "*.xlsx" -o -name "*.xls" \) -print 2>/dev/null | head -1)
            warn "⚠️  Multiple Excel files found, using first one: $(basename "$ground_truth")" >&2
        fi
    else
        info "📊 No Excel files found in lot directory - will skip metrics" >&2
    fi
    
    echo "$lot_dir|$ground_truth"
}

process_lot() {
    local lot_num="$1"
    local enable_llm="$2"
    local enable_metrics="$3"
    local output_dir="$4"
    local file_limit="$5"
    
    info "Getting lot information for LOT-$lot_num..."
    local lot_info
    lot_info=$(get_lot_info "$lot_num")
    IFS='|' read -r lot_dir ground_truth <<< "$lot_info"
    
    info "Processing LOT-$lot_num..."
    info "  📁 PDF Directory: $lot_dir"
    info "  📊 Ground Truth: ${ground_truth:-"None (auto-detect or skip metrics)"}"
    info "  📁 Output Directory: $output_dir"
    info "  🤖 LLM Approach: $([ "$enable_llm" = true ] && echo "Enabled" || echo "Disabled")"
    info "  📈 Metrics: $([ "$enable_metrics" = true ] && echo "Enabled" || echo "Disabled")"
    
    # Count PDF files
    local pdf_count
    pdf_count=$(find "$lot_dir" -name "*.pdf" | wc -l)
    info "  📄 Found $pdf_count PDF files"
    
    if [[ $pdf_count -eq 0 ]]; then
        warn "No PDF files found in $lot_dir, skipping..."
        return 0
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Check if integrated backend is running
    check_backend_running() {
        if command -v curl >/dev/null 2>&1; then
            if curl -s "$INTEGRATED_BACKEND_URL/api/services/health" >/dev/null 2>&1; then
                return 0
            fi
        fi
        return 1
    }
    
    # Start backend if not running
    if ! check_backend_running; then
        warn "Integrated backend is not running. Starting it..."
        "$BACKEND_SCRIPT" --force >/dev/null 2>&1 &
        
        # Wait for backend to start
        local retries=0
        while ! check_backend_running && [ $retries -lt 30 ]; do
            sleep 2
            retries=$((retries + 1))
        done
        
        if ! check_backend_running; then
            error "Failed to start integrated backend after 60 seconds"
            return 1
        fi
        
        info "✅ Integrated backend started successfully"
    else
        info "✅ Integrated backend is already running"
    fi
    
    # Call integrated backend API for batch processing
    (
        # Build JSON payload
        local payload="{"
        payload="$payload\"pdf_folder\": \"$lot_dir\""
        
        # Add ground truth if available
        if [[ -n "$ground_truth" && -f "$ground_truth" ]]; then
            payload="$payload, \"ground_truth_file\": \"$ground_truth\""
        fi
        
        # Add output folder
        payload="$payload, \"output_folder\": \"$output_dir\""
        
        # Add options
        payload="$payload, \"options\": {"
        
        # Add approaches (Pure LLM or Hybrid RAG only)
        if [[ "$enable_llm" == "true" ]]; then
            payload="$payload\"approaches\": [\"hybrid_rag\", \"pure_llm\"]"
        else
            payload="$payload\"approaches\": [\"hybrid_rag\"]"
        fi
        
        # Add confidence threshold
        payload="$payload, \"confidence_threshold\": 0.3"
        
        # Add max pages
        payload="$payload, \"max_pages\": 2"
        
        payload="$payload}"
        
        # Add enable metrics
        payload="$payload, \"enable_metrics\": $enable_metrics"
        
        payload="$payload}"
        
        info "Calling integrated backend API..."
        info "Payload: $payload"
        
        # Make API call
        local response
        if command -v curl >/dev/null 2>&1; then
            response=$(curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "$payload" \
                "$INTEGRATED_BACKEND_URL/api/services/hybrid-rag-classification/process-folder")
            local curl_exit_code=$?
            
            if [[ $curl_exit_code -eq 0 ]]; then
                # Check if response indicates success
                if command -v jq >/dev/null 2>&1; then
                    local success_status=$(echo "$response" | jq -r '.success // false')
                    if [[ "$success_status" == "true" ]]; then
                        success "LOT-$lot_num processing completed successfully"
                        
                        # Extract stats from response
                        local total_files=$(echo "$response" | jq -r '.data.total_files // 0')
                        local processed_files=$(echo "$response" | jq -r '.data.processing_stats.processed_files // 0')
                        local failed_files=$(echo "$response" | jq -r '.data.processing_stats.failed_files // 0')
                        
                        info "📄 Total files: $total_files"
                        info "✅ Processed: $processed_files"
                        info "❌ Failed: $failed_files"
                        info "Results saved to: $output_dir"
                        
                        return 0
                    else
                        local error_msg=$(echo "$response" | jq -r '.error // "Unknown error"')
                        error "API returned error: $error_msg"
                        return 1
                    fi
                else
                    # Fallback without jq
                    if echo "$response" | grep -q '"success".*true'; then
                        success "LOT-$lot_num processing completed successfully"
                        info "Results saved to: $output_dir"
                        return 0
                    else
                        error "API call failed or returned error"
                        info "Response: $response"
                        return 1
                    fi
                fi
            else
                error "Failed to call integrated backend API (curl exit code: $curl_exit_code)"
                return 1
            fi
        else
            error "curl command not available - cannot call integrated backend API"
            return 1
        fi
    )
}

# Main script logic
main() {
    local lot_num=""
    local enable_llm="false"
    local enable_metrics="true"
    local output_dir=""
    local file_limit=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --with-llm)
                enable_llm="true"
                shift
                ;;
            --no-metrics)
                enable_metrics="false"
                shift
                ;;
            --output)
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    output_dir="$2"
                    shift 2
                else
                    error "--output requires a directory argument"
                    exit 1
                fi
                ;;
            --limit)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    file_limit="$2"
                    shift 2
                else
                    error "--limit requires a positive number"
                    exit 1
                fi
                ;;
            --*)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [[ -z "$lot_num" ]]; then
                    lot_num="$1"
                else
                    error "Multiple lot numbers specified. Only one allowed."
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate lot number
    if [[ -z "$lot_num" ]]; then
        error "Lot number is required"
        show_help
        exit 1
    fi
    
    if ! [[ "$lot_num" =~ ^[0-9]+$ ]] || [[ $lot_num -lt 21 || $lot_num -gt 27 ]]; then
        error "Invalid lot number: $lot_num. Must be 21-27."
        exit 1
    fi
    
    # Set default output directory if not specified
    if [[ -z "$output_dir" ]]; then
        output_dir="$RESULTS_BASE_DIR/LOT-$lot_num"
    fi
    
    # Check virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found at $VENV_DIR"
        exit 1
    fi
    
    # Process the lot
    process_lot "$lot_num" "$enable_llm" "$enable_metrics" "$output_dir" "$file_limit"
}

main "$@"