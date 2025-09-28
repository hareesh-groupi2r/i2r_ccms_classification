#!/usr/bin/env python3
"""
Unified Single PDF Processor
Process single PDFs using the unified processing pipeline (no API dependency)
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_pdf_processor import UnifiedPDFProcessor


def display_classification_results(result: Dict, show_details: bool = True):
    """Display classification results in a formatted way."""
    filename = result['file_name']
    print(f"\n📊 CLASSIFICATION RESULTS FOR: {filename}")
    print("=" * 80)
    
    if result['status'] != 'completed':
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        return
    
    # Display unified results (best across all approaches)
    unified = result['unified_results']
    
    print(f"🎯 UNIFIED RESULTS (Best across all approaches):")
    print(f"   Overall Confidence: {unified['confidence_score']:.3f}")
    print(f"   Processing Time: {result['processing_time']:.2f}s")
    print()
    
    # Display categories
    categories = unified['categories']
    print(f"📂 CATEGORIES ({len(categories)} found):")
    if categories:
        for i, category in enumerate(categories, 1):
            confidence = category.get('confidence', 0)
            source = category.get('source_approach', 'unknown')
            print(f"   {i}. {category.get('category', 'Unknown')}")
            print(f"      Confidence: {confidence:.3f} | Source: {source}")
    else:
        print("   No categories identified above threshold")
    
    print()
    
    # Display issues
    issues = unified['issues']
    print(f"🔍 ISSUES ({len(issues)} found):")
    if issues:
        for i, issue in enumerate(issues, 1):
            confidence = issue.get('confidence', 0)
            source = issue.get('source_approach', 'unknown')
            print(f"   {i}. {issue.get('issue_type', 'Unknown')}")
            print(f"      Confidence: {confidence:.3f} | Source: {source}")
    else:
        print("   No issues identified above threshold")
    
    print()
    
    # Display extraction info
    extraction = result.get('extraction_info', {})
    print(f"📄 EXTRACTION INFO:")
    print(f"   Method: {extraction.get('extraction_method', 'Unknown')}")
    print(f"   Correspondence Method: {extraction.get('correspondence_method', 'Unknown')}")
    print(f"   Raw Text Length: {extraction.get('raw_length', 0)} chars")
    print(f"   Focused Content Length: {extraction.get('focused_length', 0)} chars")
    print()
    
    # Display ground truth comparison if available
    if 'ground_truth_comparison' in result:
        gt_comparison = result['ground_truth_comparison']
        print(f"📊 GROUND TRUTH COMPARISON:")
        
        if gt_comparison['status'] == 'compared':
            print(f"   Ground Truth: {', '.join(gt_comparison['ground_truth_categories'])}")
            print(f"   Predicted: {', '.join(gt_comparison['predicted_categories'])}")
            print(f"   Precision: {gt_comparison['precision']:.3f}")
            print(f"   Recall: {gt_comparison['recall']:.3f}")
            print(f"   F1-Score: {gt_comparison['f1_score']:.3f}")
            
            if gt_comparison['missing_categories']:
                print(f"   Missing: {', '.join(gt_comparison['missing_categories'])}")
            if gt_comparison['extra_categories']:
                print(f"   Extra: {', '.join(gt_comparison['extra_categories'])}")
        else:
            print(f"   Status: {gt_comparison['status']}")
            if 'message' in gt_comparison:
                print(f"   Message: {gt_comparison['message']}")
        
        print()
    
    if show_details:
        # Display approach-specific results
        print(f"🔧 APPROACH-SPECIFIC RESULTS:")
        approaches = result.get('approaches', {})
        for approach_name, approach_data in approaches.items():
            status = approach_data.get('status', 'unknown')
            processing_time = approach_data.get('processing_time', 0)
            
            if status == 'success':
                categories_count = len(approach_data.get('categories', []))
                issues_count = len(approach_data.get('issues', []))
                provider = approach_data.get('provider_used', 'unknown')
                
                print(f"   {approach_name.replace('_', ' ').title()}: ✅ Success")
                print(f"     Categories: {categories_count} | Issues: {issues_count}")
                print(f"     Provider: {provider} | Time: {processing_time:.2f}s")
            else:
                error_type = approach_data.get('error_type', 'unknown')
                error_msg = approach_data.get('error_message', 'Unknown error')
                print(f"   {approach_name.replace('_', ' ').title()}: ❌ {error_type}")
                print(f"     Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
                print(f"     Time: {processing_time:.2f}s")
    
    print("=" * 80)


def save_results_to_json(result: Dict, output_dir: str = None) -> str:
    """Save results to JSON file."""
    if output_dir is None:
        output_dir = './results/unified_single_pdf'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    pdf_name = Path(result['file_name']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{pdf_name}_unified_result_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Results saved to JSON: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"❌ Failed to save JSON results: {e}")
        return None


def save_results_to_excel(result: Dict, output_dir: str = None) -> str:
    """Save results to Excel file."""
    if output_dir is None:
        output_dir = './results/unified_single_pdf'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    pdf_name = Path(result['file_name']).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{pdf_name}_unified_result_{timestamp}.xlsx"
    
    try:
        # Prepare unified results data
        unified_row = {
            'Filename': result['file_name'],
            'Processing_Time_Seconds': result['processing_time'],
            'Overall_Confidence': result['unified_results']['confidence_score'],
            'Categories_Found': len(result['unified_results']['categories']),
            'Issues_Found': len(result['unified_results']['issues']),
            'Categories_List': ', '.join([cat['category'] for cat in result['unified_results']['categories']]),
            'Issues_List': ', '.join([issue['issue_type'] for issue in result['unified_results']['issues']]),
            'Subject': result.get('extraction_info', {}).get('subject', ''),
            'Body': result.get('extraction_info', {}).get('body', '')
        }
        
        # Approach-specific results
        approach_rows = []
        for approach_name, approach_data in result.get('approaches', {}).items():
            if approach_data.get('status') == 'success':
                categories = approach_data.get('categories', [])
                issues = approach_data.get('issues', [])
                
                approach_row = {
                    'Filename': result['file_name'],
                    'Approach': approach_name.replace('_', ' ').title(),
                    'Status': 'Success',
                    'Categories': ', '.join([cat.get('category', '') for cat in categories]),
                    'Category_Confidences': ', '.join([f"{cat.get('confidence', 0):.3f}" for cat in categories]),
                    'Issues': ', '.join([issue.get('issue_type', '') for issue in issues]),
                    'Issue_Confidences': ', '.join([f"{issue.get('confidence', 0):.3f}" for issue in issues]),
                    'Processing_Time': approach_data.get('processing_time', 0),
                    'Provider_Used': approach_data.get('provider_used', 'unknown')
                }
            else:
                approach_row = {
                    'Filename': result['file_name'],
                    'Approach': approach_name.replace('_', ' ').title(),
                    'Status': f"Failed: {approach_data.get('error_type', 'unknown')}",
                    'Categories': '',
                    'Category_Confidences': '',
                    'Issues': '',
                    'Issue_Confidences': '',
                    'Processing_Time': approach_data.get('processing_time', 0),
                    'Provider_Used': 'N/A',
                    'Error_Message': approach_data.get('error_message', '')[:200]
                }
            
            approach_rows.append(approach_row)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Unified results sheet
            unified_df = pd.DataFrame([unified_row])
            unified_df.to_excel(writer, sheet_name='Unified Results', index=False)
            
            # Approach-specific results sheet
            if approach_rows:
                approach_df = pd.DataFrame(approach_rows)
                approach_df.to_excel(writer, sheet_name='Approach Results', index=False)
        
        print(f"💾 Results saved to Excel: {output_file}")
        return str(output_file)
    
    except Exception as e:
        print(f"❌ Failed to save Excel results: {e}")
        return None


def process_single_pdf(pdf_path: str, 
                      approaches: List[str] = None,
                      confidence_threshold: float = 0.3,
                      max_pages: int = None,
                      show_details: bool = True,
                      save_json: bool = True,
                      save_excel: bool = True,
                      ground_truth_file: str = None) -> bool:
    """Process a single PDF using unified processor."""
    
    print(f"🧪 UNIFIED PDF PROCESSING")
    print("=" * 80)
    print(f"File: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    
    if approaches:
        print(f"Approaches: {', '.join(approaches)}")
    if confidence_threshold != 0.3:
        print(f"Confidence Threshold: {confidence_threshold}")
    if max_pages:
        print(f"Max Pages: {max_pages}")
    
    print()
    
    try:
        # Initialize unified processor
        processor = UnifiedPDFProcessor()
        
        # Display available approaches
        available = processor.get_available_approaches()
        print(f"🔧 Available approaches: {', '.join(available)}")
        
        if approaches:
            # Validate requested approaches
            invalid = set(approaches) - set(available)
            if invalid:
                print(f"❌ Invalid approaches: {', '.join(invalid)}")
                print(f"Valid options: {', '.join(available)}")
                return False
        
        print()
        
        # Process the PDF
        result = processor.process_single_pdf(
            pdf_path=pdf_path,
            approaches=approaches,
            confidence_threshold=confidence_threshold,
            max_pages=max_pages,
            ground_truth_file=ground_truth_file
        )
        
        # Display results
        display_classification_results(result, show_details=show_details)
        
        # Save results
        json_file = None
        excel_file = None
        
        if save_json:
            json_file = save_results_to_json(result)
        
        if save_excel:
            excel_file = save_results_to_excel(result)
        
        # Summary
        success = result['status'] == 'completed'
        print(f"\n🎯 PROCESSING SUMMARY:")
        print(f"   Status: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            unified = result['unified_results']
            print(f"   Categories: {len(unified['categories'])}")
            print(f"   Issues: {len(unified['issues'])}")
            print(f"   Confidence: {unified['confidence_score']:.3f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            
            if json_file:
                print(f"   JSON Results: ✅ {json_file}")
            if excel_file:
                print(f"   Excel Results: ✅ {excel_file}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return success
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process single PDF using unified processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDF with all available approaches
  python unified_single_pdf_processor.py "data/Lot-11/filename.pdf"
  
  # Process with specific approach
  python unified_single_pdf_processor.py "data/Lot-11/filename.pdf" --approaches hybrid_rag
  
  # Process with custom confidence threshold
  python unified_single_pdf_processor.py "data/Lot-11/filename.pdf" --confidence 0.5
  
  # Process first 2 pages only
  python unified_single_pdf_processor.py "data/Lot-11/filename.pdf" --max-pages 2
  
  # Process with multiple approaches
  python unified_single_pdf_processor.py "data/Lot-11/filename.pdf" --approaches hybrid_rag pure_llm
        """
    )
    
    parser.add_argument('pdf_file', help='Path to PDF file to process')
    parser.add_argument('--approaches', nargs='+', 
                       help='Classification approaches to use (default: all available)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum number of pages to extract')
    parser.add_argument('--no-details', action='store_true',
                       help='Hide approach-specific details')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip JSON output')
    parser.add_argument('--no-excel', action='store_true',
                       help='Skip Excel output')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth Excel file for comparison')
    
    args = parser.parse_args()
    
    pdf_path = args.pdf_file
    
    # Validate file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        
        # Show available PDFs
        pdf_files = list(Path('data/Lot-11').glob('*.pdf'))
        if pdf_files:
            print(f"\nAvailable PDF files ({len(pdf_files)} found):")
            for i, pdf_file in enumerate(pdf_files[:5], 1):
                size_kb = pdf_file.stat().st_size / 1024
                print(f"  {i}. {pdf_file.name} ({size_kb:.1f} KB)")
            if len(pdf_files) > 5:
                print(f"  ... and {len(pdf_files) - 5} more files")
        
        sys.exit(1)
    
    # Validate it's a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    # Process the PDF
    success = process_single_pdf(
        pdf_path=pdf_path,
        approaches=args.approaches,
        confidence_threshold=args.confidence,
        max_pages=args.max_pages,
        show_details=not args.no_details,
        save_json=not args.no_json,
        save_excel=not args.no_excel,
        ground_truth_file=args.ground_truth
    )
    
    if success:
        print("\n🎉 UNIFIED PDF PROCESSING SUCCESSFUL!")
        print("Results are consistent with batch processing pipeline.")
    else:
        print("\n❌ UNIFIED PDF PROCESSING FAILED!")
        print("Please check the errors above.")
    
    return success


if __name__ == "__main__":
    main()