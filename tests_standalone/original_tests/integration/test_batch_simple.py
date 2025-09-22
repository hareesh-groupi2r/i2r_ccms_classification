#!/usr/bin/env python3
"""
Simple test of the batch processing system
"""

import sys
sys.path.append('.')
import logging
from batch_processor import process_lot_pdfs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_simple_batch():
    """Test simple batch processing with only Hybrid RAG"""
    
    print("🧪 Testing Simple Batch Processing")
    print("=" * 50)
    
    try:
        # Process just 2 PDFs for testing
        results = process_lot_pdfs(
            pdf_folder="data/Lot-11",
            enable_llm=False,  # Only Hybrid RAG
            enable_metrics=True,  # Auto-detect ground truth
            output_folder="results/test_batch"
        )
        
        print(f"✅ Test completed successfully!")
        print(f"📊 Files processed: {results['processing_stats']['processed_files']}")
        print(f"📊 Files failed: {results['processing_stats']['failed_files']}")
        print(f"⏱️  Total time: {results['processing_stats']['end_time'] - results['processing_stats']['start_time']}")
        
        # Show sample results
        if results['results']:
            print(f"\n📄 Sample Results:")
            for result in results['results'][:2]:  # First 2 results
                if result['status'] == 'completed':
                    file_name = result['file_name']
                    if 'hybrid_rag' in result['approaches']:
                        categories = result['approaches']['hybrid_rag']['categories']
                        processing_time = result['approaches']['hybrid_rag']['processing_time']
                        print(f"  {file_name}")
                        print(f"    Categories: {', '.join(categories) if categories else 'None'}")
                        print(f"    Time: {processing_time:.2f}s")
                        
                        # Show metrics if available
                        if 'metrics' in result['approaches']['hybrid_rag']:
                            metrics = result['approaches']['hybrid_rag']['metrics']
                            print(f"    Metrics: F1={metrics.get('f1_score', 'N/A')}, Precision={metrics.get('precision', 'N/A')}, Recall={metrics.get('recall', 'N/A')}")
                else:
                    print(f"  {result['file_name']}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Show overall metrics if available  
        if 'overall_metrics' in results:
            print(f"\n📊 Overall Metrics:")
            for approach, metrics in results['overall_metrics'].items():
                print(f"  {approach.replace('_', ' ').title()}:")
                print(f"    Micro F1: {metrics['micro_f1']:.3f}")
                print(f"    Macro F1: {metrics['macro_f1']:.3f}")
                print(f"    Exact Match: {metrics['exact_match_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_batch()
    sys.exit(0 if success else 1)