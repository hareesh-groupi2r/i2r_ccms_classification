#!/usr/bin/env python3
"""
Test the enhanced batch processor with subject/body extraction and confidence scores
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_processor import process_lot_pdfs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_enhanced_batch():
    """Test enhanced batch processing with subject/body and confidence scores."""
    
    print("🧪 Testing Enhanced Batch Processing")
    print("=" * 50)
    
    # Test with just 2 files from LOT-21
    lot21_dir = "data/Lots21-27/Lot 21 to 23/LOT-21"
    ground_truth_file = "data/Lots21-27/Lot 21 to 23/LOT-21/LOT-21.xlsx"
    output_dir = "results/LOT-21-test"
    
    if not Path(lot21_dir).exists():
        print(f"❌ LOT-21 directory not found: {lot21_dir}")
        return
    
    if not Path(ground_truth_file).exists():
        print(f"⚠️  Ground truth file not found: {ground_truth_file}")
        ground_truth_file = None
    
    try:
        # Process with ground truth and metrics enabled
        results = process_lot_pdfs(
            pdf_folder=lot21_dir,
            ground_truth_file=ground_truth_file,
            enable_llm=False,  # Only Hybrid RAG for testing
            enable_metrics=True,
            output_folder=output_dir
        )
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Files processed: {results['processing_stats']['processed_files']}")
        print(f"📊 Files failed: {results['processing_stats']['failed_files']}")
        print(f"📁 Results saved to: {output_dir}")
        
        # Show sample results
        if results['results']:
            sample_result = results['results'][0]
            if sample_result['status'] == 'completed':
                print(f"\n📄 Sample Result:")
                print(f"  File: {sample_result['file_name']}")
                print(f"  Subject: {sample_result.get('subject', 'N/A')[:100]}...")
                print(f"  Body: {sample_result.get('body', 'N/A')[:100]}...")
                
                for approach, data in sample_result['approaches'].items():
                    print(f"  {approach.replace('_', ' ').title()}:")
                    if 'category_details' in data:
                        for cat_detail in data['category_details']:
                            cat = cat_detail.get('category', '')
                            conf = cat_detail.get('confidence', 0.0)
                            print(f"    - {cat}: {conf:.3f}")
                    else:
                        print(f"    - Categories: {data.get('categories', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_batch()
    sys.exit(0 if success else 1)