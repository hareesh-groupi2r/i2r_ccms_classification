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
PYTHON_SCRIPT="process_batch_lots.py"

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
  --help, -h        Show this help message

Examples:
  $0 21                    # Process LOT-21 with default settings
  $0 24 --with-llm        # Process LOT-24 with Pure LLM enabled
  $0 22 --no-metrics      # Process LOT-22 without metrics
  $0 25 --output my_results/lot25  # Custom output directory

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
        if [[ $lot_num -eq 21 ]]; then
            ground_truth="$lot_dir/LOT-21.xlsx"
        fi
    elif [[ $lot_num -ge 24 && $lot_num -le 27 ]]; then
        lot_dir="$LOTS_BASE_DIR/Lot 24 to 27/Lot $lot_num"
    else
        error "Invalid lot number: $lot_num. Must be 21-27."
        exit 1
    fi
    
    if [[ ! -d "$lot_dir" ]]; then
        error "Lot directory not found: $lot_dir"
        exit 1
    fi
    
    echo "$lot_dir|$ground_truth"
}

process_lot() {
    local lot_num="$1"
    local enable_llm="$2"
    local enable_metrics="$3"
    local output_dir="$4"
    
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
    
    # Create Python script if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]]; then
        info "Creating Python batch processing script..."
        cat > "$SCRIPT_DIR/$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Batch processing script for individual lots
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_processor import process_lot_pdfs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Process a lot of PDFs for classification')
    parser.add_argument('--pdf-folder', required=True, help='Path to folder containing PDF files')
    parser.add_argument('--ground-truth', help='Path to ground truth Excel file (optional)')
    parser.add_argument('--output-folder', required=True, help='Output folder for results')
    parser.add_argument('--lot-name', required=True, help='Name of the lot for logging')
    parser.add_argument('--enable-llm', action='store_true', help='Enable Pure LLM approach')
    parser.add_argument('--disable-metrics', action='store_true', help='Disable metrics calculation')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting batch processing for {args.lot_name}")
    logger.info(f"📁 PDF Folder: {args.pdf_folder}")
    logger.info(f"📊 Ground Truth: {args.ground_truth or 'Auto-detect'}")
    logger.info(f"📁 Output Folder: {args.output_folder}")
    logger.info(f"🤖 LLM Approach: {'Enabled' if args.enable_llm else 'Disabled'}")
    logger.info(f"📈 Metrics: {'Disabled' if args.disable_metrics else 'Enabled'}")
    
    try:
        # Process the lot
        results = process_lot_pdfs(
            pdf_folder=args.pdf_folder,
            ground_truth_file=args.ground_truth,
            enable_llm=args.enable_llm,
            enable_metrics=not args.disable_metrics,
            output_folder=args.output_folder
        )
        
        # Log results summary
        stats = results.get('processing_stats', {})
        logger.info(f"✅ {args.lot_name} processing completed:")
        logger.info(f"   📄 Total files: {stats.get('total_files', 0)}")
        logger.info(f"   ✅ Processed: {stats.get('processed_files', 0)}")
        logger.info(f"   ❌ Failed: {stats.get('failed_files', 0)}")
        
        if 'overall_metrics' in results:
            logger.info(f"   📊 Metrics available in results")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error processing {args.lot_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
PYTHON_EOF
        chmod +x "$SCRIPT_DIR/$PYTHON_SCRIPT"
    fi
    
    # Activate virtual environment and run processing
    (
        source "$VENV_DIR/bin/activate"
        export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
        
        # Build command
        local cmd="python $PYTHON_SCRIPT --pdf-folder \"$lot_dir\" --output-folder \"$output_dir\" --lot-name \"LOT-$lot_num\""
        
        # Add ground truth if available
        if [[ -n "$ground_truth" && -f "$ground_truth" ]]; then
            cmd="$cmd --ground-truth \"$ground_truth\""
        fi
        
        # Add options
        if [[ "$enable_llm" == "true" ]]; then
            cmd="$cmd --enable-llm"
        fi
        
        if [[ "$enable_metrics" == "false" ]]; then
            cmd="$cmd --disable-metrics"
        fi
        
        # Execute processing
        info "Executing: $cmd"
        eval "$cmd"
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            success "LOT-$lot_num processing completed successfully"
            info "Results saved to: $output_dir"
        else
            error "LOT-$lot_num processing failed with exit code $exit_code"
        fi
        
        return $exit_code
    )
}

# Main script logic
main() {
    local lot_num=""
    local enable_llm="false"
    local enable_metrics="true"
    local output_dir=""
    
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
    process_lot "$lot_num" "$enable_llm" "$enable_metrics" "$output_dir"
}

main "$@"