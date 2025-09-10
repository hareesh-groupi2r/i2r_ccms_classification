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
    
    # Create Python script if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]]; then
        info "Creating Python batch processing script..."
        cat > "$SCRIPT_DIR/$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Batch processing script for individual lots with custom file limiting
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_processor import BatchPDFProcessor
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_custom_batch_config(pdf_folder, ground_truth_file, enable_llm, enable_metrics, limit_files):
    """Create a custom batch configuration with proper ground truth handling."""
    
    # Create temporary batch config with explicit ground truth handling
    config = {
        'batch_processing': {
            'enabled': True,
            'approaches': {
                'hybrid_rag': {'enabled': True, 'priority': 1},
                'pure_llm': {'enabled': enable_llm, 'priority': 2}
            },
            'evaluation': {
                'enabled': enable_metrics,
                'auto_detect_ground_truth': ground_truth_file is None,
                'ground_truth_patterns': ["EDMS*.xlsx", "LOT-*.xlsx", "ground_truth*.xlsx", "*_labels.xlsx"]
            },
            'output': {
                'results_folder': 'results',
                'save_format': 'xlsx'
            },
            'processing': {
                'max_pages_per_pdf': 2,
                'skip_on_error': True,
                'rate_limit_delay': 1  # Reduced for testing
            }
        }
    }
    
    # Add file limiting if specified
    if limit_files:
        config['batch_processing']['processing']['max_files'] = limit_files
    
    return config

def get_pdf_files(pdf_folder, limit=None):
    """Get list of PDF files, optionally limited to first N files."""
    pdf_folder = Path(pdf_folder)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    if limit:
        pdf_files = pdf_files[:limit]
        logger.info(f"📄 Limited to first {limit} files out of {len(sorted(pdf_folder.glob('*.pdf')))} total")
    
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description='Process a lot of PDFs for classification')
    parser.add_argument('--pdf-folder', required=True, help='Path to folder containing PDF files')
    parser.add_argument('--ground-truth', help='Path to ground truth Excel file (optional)')
    parser.add_argument('--output-folder', required=True, help='Output folder for results')
    parser.add_argument('--lot-name', required=True, help='Name of the lot for logging')
    parser.add_argument('--enable-llm', action='store_true', help='Enable Pure LLM approach')
    parser.add_argument('--disable-metrics', action='store_true', help='Disable metrics calculation')
    parser.add_argument('--limit', type=int, help='Process only first N files')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting batch processing for {args.lot_name}")
    logger.info(f"📁 PDF Folder: {args.pdf_folder}")
    logger.info(f"📊 Ground Truth: {args.ground_truth or 'Auto-detect'}")
    logger.info(f"📁 Output Folder: {args.output_folder}")
    logger.info(f"🤖 LLM Approach: {'Enabled' if args.enable_llm else 'Disabled (Hybrid RAG only)'}")
    logger.info(f"📈 Metrics: {'Disabled' if args.disable_metrics else 'Enabled'}")
    if args.limit:
        logger.info(f"📄 File Limit: First {args.limit} files")
    
    try:
        # Check if ground truth file exists
        if args.ground_truth and not Path(args.ground_truth).exists():
            logger.warning(f"⚠️  Ground truth file not found: {args.ground_truth}")
            logger.info(f"📊 Will attempt auto-detection in PDF folder")
            args.ground_truth = None
        
        # Get PDF files with optional limit
        pdf_files = get_pdf_files(args.pdf_folder, args.limit)
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {args.pdf_folder}")
            return 1
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Create custom batch configuration
        batch_config = create_custom_batch_config(
            args.pdf_folder, 
            args.ground_truth, 
            args.enable_llm, 
            not args.disable_metrics,
            args.limit
        )
        
        # Save temporary config
        temp_config_path = Path("temp_single_lot_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(batch_config, f, default_flow_style=False)
        
        # Initialize batch processor with custom config
        processor = BatchPDFProcessor(
            config_path="config.yaml",
            batch_config_path=str(temp_config_path)
        )
        
        # Create a temporary folder with limited files if needed
        actual_pdf_folder = args.pdf_folder
        temp_folder = None
        
        if args.limit:
            import shutil
            temp_folder = Path(f"temp_limited_lot_{args.lot_name}")
            temp_folder.mkdir(exist_ok=True)
            
            # Copy only the first N PDF files
            pdf_files = get_pdf_files(args.pdf_folder, args.limit)
            for pdf_file in pdf_files:
                shutil.copy2(pdf_file, temp_folder / pdf_file.name)
            
            # Copy ground truth file if it exists in the original folder
            if args.ground_truth and Path(args.ground_truth).exists():
                shutil.copy2(args.ground_truth, temp_folder / Path(args.ground_truth).name)
                args.ground_truth = str(temp_folder / Path(args.ground_truth).name)
            
            actual_pdf_folder = str(temp_folder)
            logger.info(f"📄 Created temporary folder with {len(pdf_files)} files: {actual_pdf_folder}")
        
        # Process the lot
        logger.info(f"🔄 Processing {args.lot_name}...")
        results = processor.process_pdf_folder(
            pdf_folder=actual_pdf_folder,
            ground_truth_file=args.ground_truth,
            output_folder=args.output_folder
        )
        
        # Clean up temporary folder if created
        if temp_folder and temp_folder.exists():
            shutil.rmtree(temp_folder)
            logger.info(f"🧹 Cleaned up temporary folder")
        
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()
        
        # Log results summary
        stats = results.get('processing_stats', {})
        logger.info(f"✅ {args.lot_name} processing completed:")
        logger.info(f"   📄 Total files: {stats.get('total_files', 0)}")
        logger.info(f"   ✅ Processed: {stats.get('processed_files', 0)}")
        logger.info(f"   ❌ Failed: {stats.get('failed_files', 0)}")
        
        if 'overall_metrics' in results:
            metrics = results['overall_metrics']
            for approach, metric_data in metrics.items():
                logger.info(f"   📊 {approach.replace('_', ' ').title()} Metrics:")
                logger.info(f"      F1-Score: {metric_data.get('micro_f1', 'N/A'):.3f}")
                logger.info(f"      Precision: {metric_data.get('micro_precision', 'N/A'):.3f}")
                logger.info(f"      Recall: {metric_data.get('micro_recall', 'N/A'):.3f}")
        
        logger.info(f"📁 Results saved to: {args.output_folder}")
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
        
        if [[ -n "$file_limit" ]]; then
            cmd="$cmd --limit $file_limit"
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