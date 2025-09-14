#!/usr/bin/env python3
"""
Unified Batch PDF Processor
Process multiple PDFs using the unified processing pipeline
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_pdf_processor import UnifiedPDFProcessor


def load_batch_config(config_path: str = None) -> dict:
    """Load batch processing configuration."""
    if config_path is None:
        config_path = "batch_config.yaml"
    
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Return default configuration
        return {
            'batch_processing': {
                'enabled': True,
                'approaches': {
                    'hybrid_rag': {'enabled': True, 'priority': 1},
                    'pure_llm': {'enabled': False, 'priority': 2}
                },
                'evaluation': {
                    'enabled': True,
                    'auto_detect_ground_truth': True,
                    'ground_truth_patterns': ["EDMS*.xlsx", "LOT-*.xlsx", "ground_truth*.xlsx", "*_labels.xlsx"]
                },
                'output': {
                    'results_folder': 'results',
                    'save_format': 'xlsx'
                },
                'processing': {
                    'max_pages_per_pdf': 2,
                    'confidence_threshold': 0.3,
                    'skip_on_error': True,
                    'rate_limit_delay': 1
                }
            }
        }


def get_enabled_approaches_from_batch_config(batch_config: dict) -> list:
    """Get enabled approaches from batch configuration."""
    approaches_config = batch_config.get('batch_processing', {}).get('approaches', {})
    enabled_approaches = []
    
    # Sort by priority
    sorted_approaches = sorted(
        approaches_config.items(),
        key=lambda x: x[1].get('priority', 999)
    )
    
    for approach_name, config in sorted_approaches:
        if config.get('enabled', False):
            enabled_approaches.append(approach_name)
    
    return enabled_approaches


def process_batch_pdfs(pdf_folder: str,
                      batch_config_path: str = None,
                      ground_truth_file: str = None,
                      output_folder: str = None,
                      approaches: list = None,
                      confidence_threshold: float = None,
                      max_pages: int = None) -> dict:
    """
    Process multiple PDFs using unified pipeline.
    
    Args:
        pdf_folder: Path to folder containing PDF files
        batch_config_path: Path to batch configuration file
        ground_truth_file: Optional path to ground truth Excel file
        output_folder: Output folder for results
        approaches: Override approaches to use
        confidence_threshold: Override confidence threshold
        max_pages: Override max pages per PDF
        
    Returns:
        Batch processing results
    """
    # Load batch configuration
    batch_config = load_batch_config(batch_config_path)
    processing_config = batch_config['batch_processing']['processing']
    
    # Use overrides if provided
    if approaches is None:
        approaches = get_enabled_approaches_from_batch_config(batch_config)
    
    if confidence_threshold is None:
        confidence_threshold = processing_config.get('confidence_threshold', 0.3)
    
    if max_pages is None:
        max_pages = processing_config.get('max_pages_per_pdf', 2)
    
    if output_folder is None:
        output_folder = batch_config['batch_processing']['output']['results_folder']
    
    print(f"🚀 UNIFIED BATCH PDF PROCESSING")
    print("=" * 80)
    print(f"PDF Folder: {pdf_folder}")
    print(f"Approaches: {', '.join(approaches) if approaches else 'None'}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Max Pages per PDF: {max_pages}")
    print(f"Output Folder: {output_folder}")
    print(f"Ground Truth: {ground_truth_file or 'None'}")
    print()
    
    # Initialize unified processor
    try:
        processor = UnifiedPDFProcessor()
        
        # Validate approaches
        available_approaches = processor.get_available_approaches()
        print(f"🔧 Available approaches: {', '.join(available_approaches)}")
        
        if approaches:
            invalid_approaches = set(approaches) - set(available_approaches)
            if invalid_approaches:
                print(f"❌ Invalid approaches: {', '.join(invalid_approaches)}")
                return {'status': 'error', 'error': f'Invalid approaches: {invalid_approaches}'}
        else:
            approaches = available_approaches
            print(f"📋 Using all available approaches: {', '.join(approaches)}")
        
        print()
        
        # Process batch
        results = processor.process_batch_pdfs(
            pdf_folder=pdf_folder,
            approaches=approaches,
            confidence_threshold=confidence_threshold,
            max_pages=max_pages,
            ground_truth_file=ground_truth_file,
            output_folder=output_folder
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process multiple PDFs using unified processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process folder with default configuration
  python unified_batch_processor.py "data/Lot-11"
  
  # Process with specific approaches
  python unified_batch_processor.py "data/Lot-11" --approaches hybrid_rag pure_llm
  
  # Process with custom settings
  python unified_batch_processor.py "data/Lot-11" --confidence 0.5 --max-pages 5
  
  # Process with ground truth evaluation
  python unified_batch_processor.py "data/Lot-11" --ground-truth "ground_truth.xlsx"
  
  # Process with custom output folder
  python unified_batch_processor.py "data/Lot-11" --output "results/my_batch"
        """
    )
    
    parser.add_argument('pdf_folder', help='Path to folder containing PDF files')
    parser.add_argument('--config', help='Path to batch configuration file')
    parser.add_argument('--approaches', nargs='+',
                       help='Classification approaches to use')
    parser.add_argument('--confidence', type=float,
                       help='Minimum confidence threshold')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum pages to extract per PDF')
    parser.add_argument('--ground-truth', 
                       help='Path to ground truth Excel file for evaluation')
    parser.add_argument('--output',
                       help='Output folder for results')
    
    args = parser.parse_args()
    
    pdf_folder = args.pdf_folder
    
    # Validate PDF folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ PDF folder not found: {pdf_folder}")
        sys.exit(1)
    
    if not os.path.isdir(pdf_folder):
        print(f"❌ Path is not a directory: {pdf_folder}")
        sys.exit(1)
    
    # Check for PDF files
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdf_folder}")
        sys.exit(1)
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    print()
    
    # Process batch
    results = process_batch_pdfs(
        pdf_folder=pdf_folder,
        batch_config_path=args.config,
        ground_truth_file=args.ground_truth,
        output_folder=args.output,
        approaches=args.approaches,
        confidence_threshold=args.confidence,
        max_pages=args.max_pages
    )
    
    # Display summary
    if results.get('status') == 'error':
        print(f"\n❌ BATCH PROCESSING FAILED!")
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print(f"\n🎉 UNIFIED BATCH PROCESSING COMPLETED!")
        print(f"📊 Summary:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Processed: {results['processed_files']}")
        print(f"   Failed: {results['failed_files']}")
        print(f"   Success rate: {results['processed_files']/results['total_files']*100:.1f}%")
        print(f"   Total time: {results['processing_stats']['total_processing_time']:.2f}s")
        print(f"   Avg time per file: {results['processing_stats']['total_processing_time']/results['total_files']:.2f}s")
        
        print(f"\n📁 Results are consistent with single file processing pipeline.")
        return True


if __name__ == "__main__":
    main()