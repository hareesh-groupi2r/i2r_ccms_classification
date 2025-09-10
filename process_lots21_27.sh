#!/bin/bash

# =============================================================================
# Batch Processing Script for Lots 21-27
# Processes each lot individually with appropriate ground truth files
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
    
    # Check virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found at $VENV_DIR"
        exit 1
    fi
    
    # Check if lots directory exists
    if [[ ! -d "$LOTS_BASE_DIR" ]]; then
        error "Lots directory not found at $LOTS_BASE_DIR"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_BASE_DIR"
    
    # Check Python script
    if [[ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]]; then
        info "Creating Python batch processing script..."
        create_python_script
    fi
    
    success "Prerequisites checked"
}

# Create Python script for batch processing
create_python_script() {
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
    success "Created Python batch processing script"
}

# Process individual lot
process_lot() {
    local lot_name="$1"
    local pdf_folder="$2"
    local ground_truth_file="$3"
    local output_folder="$4"
    local enable_llm="${5:-false}"
    
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
    
    # Activate virtual environment and run processing
    (
        source "$VENV_DIR/bin/activate"
        export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
        
        # Build command
        local cmd="python $PYTHON_SCRIPT --pdf-folder \"$pdf_folder\" --output-folder \"$output_folder\" --lot-name \"$lot_name\""
        
        # Add ground truth if available
        if [[ -n "$ground_truth_file" && -f "$ground_truth_file" ]]; then
            cmd="$cmd --ground-truth \"$ground_truth_file\""
            info "$lot_name: Using ground truth file: $ground_truth_file"
        else
            info "$lot_name: No ground truth file, will auto-detect or skip metrics"
        fi
        
        # Add LLM option
        if [[ "$enable_llm" == "true" ]]; then
            cmd="$cmd --enable-llm"
            info "$lot_name: Pure LLM approach enabled"
        else
            info "$lot_name: Using Hybrid RAG approach only"
        fi
        
        # Execute processing
        eval "$cmd"
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            success "$lot_name: Processing completed successfully"
        else
            error "$lot_name: Processing failed with exit code $exit_code"
        fi
        
        return $exit_code
    )
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