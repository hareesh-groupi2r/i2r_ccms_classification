#!/usr/bin/env python3
"""
Example Usage of Batch PDF Processor
Demonstrates different ways to use the batch processing system
"""

import logging
from pathlib import Path
from batch_processor import BatchPDFProcessor, process_lot_pdfs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def example_1_simple_processing():
    """Example 1: Simple processing with convenience function"""
    print("🔧 Example 1: Simple Lot Processing")
    print("=" * 50)
    
    # Process Lot-11 PDFs with default settings
    # - Hybrid RAG enabled (default)
    # - Pure LLM disabled (default)
    # - Auto-detect ground truth
    # - Metrics enabled
    
    results = process_lot_pdfs(
        pdf_folder="data/Lot-11",
        enable_llm=False,  # Only Hybrid RAG
        enable_metrics=True,  # Auto-detect EDMS*.xlsx
        output_folder="results/example1"
    )
    
    print(f"✅ Processed {results['processing_stats']['processed_files']} files")
    print(f"📊 Results saved to: results/example1/")
    return results

def example_2_both_approaches():
    """Example 2: Run both approaches for comparison"""
    print("\n🔧 Example 2: Both Approaches Comparison")
    print("=" * 50)
    
    # Process with both approaches enabled
    results = process_lot_pdfs(
        pdf_folder="data/Lot-11",
        ground_truth_file="data/raw/EDMS-Lot 11.xlsx",  # Explicit ground truth
        enable_llm=True,  # Enable Pure LLM
        enable_metrics=True,
        output_folder="results/example2"
    )
    
    print(f"✅ Processed {results['processing_stats']['processed_files']} files")
    
    # Print comparison if metrics are available
    if 'overall_metrics' in results:
        print("\n📊 Approach Comparison:")
        for approach, metrics in results['overall_metrics'].items():
            print(f"  {approach.replace('_', ' ').title()}: F1={metrics['micro_f1']:.3f}, Precision={metrics['micro_precision']:.3f}, Recall={metrics['micro_recall']:.3f}")
    
    return results

def example_3_advanced_configuration():
    """Example 3: Advanced configuration with custom batch processor"""
    print("\n🔧 Example 3: Advanced Configuration")
    print("=" * 50)
    
    # Create custom batch configuration
    from pathlib import Path
    import yaml
    
    custom_config = {
        'batch_processing': {
            'enabled': True,
            'approaches': {
                'hybrid_rag': {'enabled': True, 'priority': 1},
                'pure_llm': {'enabled': True, 'priority': 2}
            },
            'evaluation': {
                'enabled': True,
                'auto_detect_ground_truth': True,
                'ground_truth_patterns': ["EDMS*.xlsx", "*.xlsx"]
            },
            'output': {
                'results_folder': 'results/example3',
                'save_format': 'xlsx',
                'include_confidence_scores': True
            },
            'processing': {
                'max_pages_per_pdf': 3,  # Process 3 pages instead of 2
                'skip_on_error': True,
                'rate_limit_delay': 1  # Faster processing
            },
            'reporting': {
                'generate_summary': True,
                'include_performance_metrics': True
            }
        }
    }
    
    # Save custom configuration
    config_path = Path("custom_batch_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f)
    
    try:
        # Initialize processor with custom configuration
        processor = BatchPDFProcessor(batch_config_path=str(config_path))
        
        # Process PDFs
        results = processor.process_pdf_folder(
            pdf_folder="data/Lot-11",
            output_folder="results/example3"
        )
        
        print(f"✅ Advanced processing completed")
        print(f"📊 Total time: {results['processing_stats']['end_time'] - results['processing_stats']['start_time']}")
        
        return results
        
    finally:
        # Cleanup custom config
        if config_path.exists():
            config_path.unlink()

def example_4_inference_mode():
    """Example 4: Inference mode without ground truth"""
    print("\n🔧 Example 4: Inference Mode (No Ground Truth)")
    print("=" * 50)
    
    # Process PDFs without ground truth (pure inference)
    # This is useful when you want to classify new documents
    
    results = process_lot_pdfs(
        pdf_folder="data/Lot-11",
        ground_truth_file=None,  # No ground truth
        enable_llm=False,  # Only Hybrid RAG for speed
        enable_metrics=False,  # Disable metrics since no ground truth
        output_folder="results/inference"
    )
    
    print(f"✅ Inference completed for {results['processing_stats']['processed_files']} files")
    print("📊 No metrics calculated (inference mode)")
    
    # Show sample predictions
    if results['results']:
        print("\n🔍 Sample Predictions:")
        for result in results['results'][:3]:  # First 3 files
            if result['status'] == 'completed' and 'hybrid_rag' in result['approaches']:
                categories = result['approaches']['hybrid_rag']['categories']
                print(f"  {result['file_name']}: {', '.join(categories) if categories else 'No categories'}")
    
    return results

def example_5_error_handling():
    """Example 5: Error handling and recovery"""
    print("\n🔧 Example 5: Error Handling")
    print("=" * 50)
    
    # Process with a folder that might have some problematic files
    try:
        results = process_lot_pdfs(
            pdf_folder="data/Lot-11",  # Some files might fail
            enable_llm=True,
            enable_metrics=True,
            output_folder="results/error_handling"
        )
        
        print(f"✅ Processing completed:")
        print(f"   Successful: {results['processing_stats']['processed_files']}")
        print(f"   Failed: {results['processing_stats']['failed_files']}")
        print(f"   Total: {results['processing_stats']['total_files']}")
        
        # Show failed files if any
        failed_files = [r for r in results['results'] if r['status'] == 'failed']
        if failed_files:
            print(f"\n❌ Failed Files:")
            for failed_result in failed_files:
                print(f"   {failed_result['file_name']}: {failed_result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return None

def main():
    """Run all examples"""
    print("🚀 Batch PDF Processing Examples")
    print("=" * 60)
    
    # Check if data directory exists
    data_path = Path("data/Lot-11")
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please ensure the PDF files are in the correct location.")
        return
    
    try:
        # Run examples
        example_1_simple_processing()
        example_2_both_approaches() 
        example_3_advanced_configuration()
        example_4_inference_mode()
        example_5_error_handling()
        
        print("\n🎉 All examples completed successfully!")
        print("📁 Check the results/ directory for outputs")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()