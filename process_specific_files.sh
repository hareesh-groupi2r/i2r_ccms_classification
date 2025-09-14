#!/bin/bash

# =============================================================================
# Specific Files Processing Script
# Process specific PDF files in an isolated manner for testing/debugging
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
LOTS_BASE_DIR="$SCRIPT_DIR/data/Lots21-27"
RESULTS_BASE_DIR="$SCRIPT_DIR/results"
TEMP_DIR="$SCRIPT_DIR/temp_specific_files"

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

cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
        info "🧹 Cleaned up temporary directory"
    fi
}

trap cleanup EXIT

show_help() {
    cat << EOF
Specific Files Processing Script

Usage: $0 LOT_NUMBER [OPTIONS]

Arguments:
  LOT_NUMBER        The lot number (21-27) to search for files in

Options:
  --files FILE1,FILE2,...  Comma-separated list of specific PDF filenames
  --pattern PATTERN        Shell pattern to match files (e.g., "*AE_SPK*", "*Change*")
  --list                   List all available PDF files in the lot and exit
  --limit N                Process only first N files (works with --pattern)
  --with-llm              Enable Pure LLM approach
  --no-metrics            Disable metrics calculation  
  --output DIR            Custom output directory
  --help, -h              Show this help message

File Selection (mutually exclusive):
  --files     : Process specific named files
  --pattern   : Process files matching a shell pattern
  --limit     : Process first N files (from pattern or all files)

Examples:
  # List all files in LOT-21
  $0 21 --list
  
  # Process specific files by name
  $0 21 --files "file1.pdf,file2.pdf"
  
  # Process files matching a pattern
  $0 21 --pattern "*AE_SPK*"
  
  # Process files with "Change" in name
  $0 21 --pattern "*Change*"
  
  # Process first 3 files matching pattern
  $0 21 --pattern "*2020*" --limit 3
  
  # Process specific files with LLM enabled
  $0 21 --files "important_file.pdf" --with-llm

Pattern Examples:
  "*AE_SPK*"           - Files containing "AE_SPK" 
  "*Change*"           - Files containing "Change"
  "*2020*"             - Files containing "2020"
  "202012*.pdf"        - Files starting with "202012"
  "*reminder*.pdf"     - Files containing "reminder"

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
            echo "  LOT-$lot: $pdf_count PDFs"
        else
            echo "  LOT-$lot: directory not found"
        fi
    done
}

get_lot_directory() {
    local lot_num="$1"
    local lot_dir=""
    
    # Determine lot directory based on lot number
    if [[ $lot_num -ge 21 && $lot_num -le 23 ]]; then
        lot_dir="$LOTS_BASE_DIR/Lot 21 to 23/LOT-$lot_num"
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
    
    echo "$lot_dir"
}

list_files() {
    local lot_num="$1"
    local lot_dir
    lot_dir=$(get_lot_directory "$lot_num")
    
    info "📁 Listing all PDF files in LOT-$lot_num:"
    info "   Directory: $lot_dir"
    echo
    
    local pdf_files
    mapfile -t pdf_files < <(find "$lot_dir" -name "*.pdf" -type f | sort)
    
    if [[ ${#pdf_files[@]} -eq 0 ]]; then
        warn "No PDF files found in LOT-$lot_num"
        return 1
    fi
    
    echo "Found ${#pdf_files[@]} PDF files:"
    local count=1
    for file in "${pdf_files[@]}"; do
        local filename
        filename=$(basename "$file")
        local size_kb
        size_kb=$(du -k "$file" | cut -f1)
        printf "%3d. %-60s (%d KB)\n" "$count" "$filename" "$size_kb"
        ((count++))
    done
    
    echo
    success "Listed ${#pdf_files[@]} files from LOT-$lot_num"
}

find_specific_files() {
    local lot_num="$1"
    local files_list="$2"
    local lot_dir
    lot_dir=$(get_lot_directory "$lot_num")
    
    info "🔍 Finding specific files in LOT-$lot_num..."
    
    # Split comma-separated file list
    IFS=',' read -ra file_names <<< "$files_list"
    local found_files=()
    local missing_files=()
    
    for file_name in "${file_names[@]}"; do
        # Remove leading/trailing whitespace
        file_name=$(echo "$file_name" | xargs)
        
        # Try to find the file (case-insensitive)
        local found_file
        found_file=$(find "$lot_dir" -iname "$file_name" -type f | head -1)
        
        if [[ -n "$found_file" ]]; then
            found_files+=("$found_file")
            info "  ✅ Found: $file_name"
        else
            missing_files+=("$file_name")
            warn "  ❌ Not found: $file_name"
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        error "Missing files: ${missing_files[*]}"
        info "💡 Use --list to see all available files"
        
        # Show similar file names
        info "🔍 Files with similar names:"
        for missing in "${missing_files[@]}"; do
            # Extract key words from missing filename
            local key_word
            key_word=$(echo "$missing" | sed 's/\.pdf//i' | sed 's/[_-]/ /g' | awk '{print $1}')
            if [[ -n "$key_word" && ${#key_word} -gt 2 ]]; then
                local similar_files
                mapfile -t similar_files < <(find "$lot_dir" -iname "*$key_word*" -type f | head -3)
                if [[ ${#similar_files[@]} -gt 0 ]]; then
                    echo "  Similar to '$missing':"
                    for similar in "${similar_files[@]}"; do
                        echo "    - $(basename "$similar")"
                    done
                fi
            fi
        done
        
        if [[ ${#found_files[@]} -eq 0 ]]; then
            exit 1
        fi
    fi
    
    # Return found files as space-separated string
    printf '%s\n' "${found_files[@]}"
}

find_pattern_files() {
    local lot_num="$1"
    local pattern="$2"
    local limit="${3:-}"
    local lot_dir
    lot_dir=$(get_lot_directory "$lot_num")
    
    info "🔍 Finding files matching pattern '$pattern' in LOT-$lot_num..."
    
    local pattern_files
    mapfile -t pattern_files < <(find "$lot_dir" -name "$pattern" -type f | sort)
    
    if [[ ${#pattern_files[@]} -eq 0 ]]; then
        error "No files match pattern: $pattern"
        info "💡 Use --list to see all available files"
        exit 1
    fi
    
    info "  📄 Found ${#pattern_files[@]} files matching pattern"
    
    # Apply limit if specified
    if [[ -n "$limit" && "$limit" -gt 0 ]]; then
        if [[ "$limit" -lt ${#pattern_files[@]} ]]; then
            pattern_files=("${pattern_files[@]:0:$limit}")
            info "  📊 Limited to first $limit files"
        fi
    fi
    
    # Show matched files
    local count=1
    for file in "${pattern_files[@]}"; do
        local filename
        filename=$(basename "$file")
        echo "  $count. $filename"
        ((count++))
    done
    
    # Return found files as space-separated string
    printf '%s\n' "${pattern_files[@]}"
}

create_temp_processing_folder() {
    local files_array=("$@")
    
    # Clean up any existing temp directory
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
    
    # Create temp directory
    mkdir -p "$TEMP_DIR"
    info "📁 Created temporary processing directory: $TEMP_DIR"
    
    # Copy selected files to temp directory
    local count=0
    for file in "${files_array[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$TEMP_DIR/"
            ((count++))
            info "  📋 Copied: $(basename "$file")"
        fi
    done
    
    success "📦 Prepared $count files for processing"
    echo "$TEMP_DIR"
}

process_specific_files() {
    local lot_num="$1"
    local files_to_process=("$@")
    shift
    local enable_llm="$1"
    local enable_metrics="$2"
    local output_dir="$3"
    
    info "🚀 Processing ${#files_to_process[@]} specific files from LOT-$lot_num..."
    
    # Create temporary processing folder
    local temp_folder
    temp_folder=$(create_temp_processing_folder "${files_to_process[@]}")
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Use the unified batch processor for consistency
    (
        source "$VENV_DIR/bin/activate"
        export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
        
        # Build the command using unified processor
        local cmd="python unified_batch_processor.py \"$temp_folder\""
        
        # Add output option
        cmd="$cmd --output \"$output_dir\""
        
        # Add approaches based on options
        if [[ "$enable_llm" == "true" ]]; then
            cmd="$cmd --approaches hybrid_rag pure_llm"
        else
            cmd="$cmd --approaches hybrid_rag"
        fi
        
        # Add confidence threshold
        cmd="$cmd --confidence 0.3"
        
        # Add max pages (consistent with lot processing)
        cmd="$cmd --max-pages 2"
        
        # Execute processing
        info "🔄 Executing: $cmd"
        eval "$cmd"
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            success "✅ Specific files processing completed successfully"
            info "📁 Results saved to: $output_dir"
            
            # List the output files for user convenience
            if [[ -d "$output_dir" ]]; then
                info "📊 Generated files:"
                find "$output_dir" -name "*.xlsx" -o -name "*.json" | while read -r result_file; do
                    echo "  📋 $(basename "$result_file")"
                done
            fi
        else
            error "❌ Specific files processing failed with exit code $exit_code"
        fi
        
        return $exit_code
    )
}

# Main script logic
main() {
    local lot_num=""
    local files_list=""
    local pattern=""
    local list_only="false"
    local limit=""
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
            --files)
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    files_list="$2"
                    shift 2
                else
                    error "--files requires a comma-separated list of filenames"
                    exit 1
                fi
                ;;
            --pattern)
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    pattern="$2"
                    shift 2
                else
                    error "--pattern requires a shell pattern argument"
                    exit 1
                fi
                ;;
            --list)
                list_only="true"
                shift
                ;;
            --limit)
                if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                    limit="$2"
                    shift 2
                else
                    error "--limit requires a positive number"
                    exit 1
                fi
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
    
    # Check virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found at $VENV_DIR"
        exit 1
    fi
    
    # Handle list-only mode
    if [[ "$list_only" == "true" ]]; then
        list_files "$lot_num"
        exit 0
    fi
    
    # Validate file selection options (mutually exclusive)
    local selection_count=0
    [[ -n "$files_list" ]] && ((selection_count++))
    [[ -n "$pattern" ]] && ((selection_count++))
    
    if [[ $selection_count -eq 0 ]]; then
        error "Must specify either --files or --pattern"
        show_help
        exit 1
    fi
    
    if [[ $selection_count -gt 1 ]]; then
        error "Cannot use --files and --pattern together"
        exit 1
    fi
    
    # Set default output directory if not specified
    if [[ -z "$output_dir" ]]; then
        local timestamp
        timestamp=$(date +"%Y%m%d_%H%M%S")
        output_dir="$RESULTS_BASE_DIR/specific_files_LOT${lot_num}_$timestamp"
    fi
    
    # Find files to process
    local files_to_process=()
    if [[ -n "$files_list" ]]; then
        info "🔍 Processing specific files: $files_list"
        mapfile -t files_to_process < <(find_specific_files "$lot_num" "$files_list")
    elif [[ -n "$pattern" ]]; then
        info "🔍 Processing files matching pattern: $pattern"
        mapfile -t files_to_process < <(find_pattern_files "$lot_num" "$pattern" "$limit")
    fi
    
    if [[ ${#files_to_process[@]} -eq 0 ]]; then
        error "No files selected for processing"
        exit 1
    fi
    
    info "📊 Selected ${#files_to_process[@]} files for processing:"
    for file in "${files_to_process[@]}"; do
        echo "  📄 $(basename "$file")"
    done
    echo
    
    # Process the files
    process_specific_files "$lot_num" "${files_to_process[@]}" "$enable_llm" "$enable_metrics" "$output_dir"
}

main "$@"