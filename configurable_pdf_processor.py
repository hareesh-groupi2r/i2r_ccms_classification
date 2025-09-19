#!/usr/bin/env python3
"""
PDF processor with configurable page limits for extraction
"""

import sys
import os
from pathlib import Path
import requests
import pandas as pd
import time
import json
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.pdf_extractor import PDFExtractor


def extract_pdf_content_with_page_limit(pdf_path: str, max_pages: int = None) -> dict:
    """Extract content from PDF file with configurable page limit."""
    print(f"📄 Extracting content from: {Path(pdf_path).name}")
    
    if max_pages:
        print(f"   🔢 Page limit: First {max_pages} pages only")
    else:
        print(f"   🔢 Page limit: All pages")
    
    start_time = time.time()
    
    try:
        # Initialize PDF extractor with page limit
        pdf_extractor = PDFExtractor(max_pages=max_pages)
        
        # extract_text returns (text, method) tuple
        text, method = pdf_extractor.extract_text(pdf_path)
        
        extraction_time = time.time() - start_time
        
        if text and text.strip():
            # Get metadata separately
            metadata = pdf_extractor.extract_metadata(pdf_path)
            
            print(f"✅ Extraction successful:")
            print(f"   Method: {method}")
            print(f"   Total pages in PDF: {metadata.get('page_count', 0)}")
            print(f"   Pages processed: {max_pages if max_pages and max_pages < metadata.get('page_count', 0) else metadata.get('page_count', 0)}")
            print(f"   Characters extracted: {len(text)}")
            print(f"   Extraction time: {extraction_time:.2f}s")
            print(f"   Preview: {text[:200]}...")
            
            return {
                'success': True,
                'text': text.strip(),
                'metadata': metadata,
                'method': method,
                'extraction_time': extraction_time,
                'pages_processed': max_pages if max_pages and max_pages < metadata.get('page_count', 0) else metadata.get('page_count', 0),
                'total_pages': metadata.get('page_count', 0),
                'page_limit_applied': max_pages is not None and max_pages < metadata.get('page_count', 0)
            }
        else:
            print(f"❌ Extraction failed: No text extracted using method: {method}")
            return {
                'success': False, 
                'error': f'No text extracted using method: {method}',
                'extraction_time': extraction_time
            }
            
    except Exception as e:
        extraction_time = time.time() - start_time
        print(f"❌ Extraction error: {e}")
        return {
            'success': False, 
            'error': str(e),
            'extraction_time': extraction_time
        }


def classify_with_api(text: str, filename: str) -> dict:
    """Classify text using the production API."""
    print(f"🔍 Classifying document...")
    
    # Truncate text if too long (API limit)
    max_length = 8000
    if len(text) > max_length:
        text = text[:max_length] + "..."
        print(f"   Text truncated to {max_length} characters")
    
    payload = {
        "text": text,
        "approach": "hybrid_rag",  # Use hybrid_rag as it's more reliable
        "confidence_threshold": 0.3,
        "max_results": 7
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8000/classify", 
            json=payload, 
            timeout=60
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result['total_processing_time'] = processing_time
            
            print(f"✅ Classification successful:")
            print(f"   API processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Total time: {processing_time:.2f}s")
            print(f"   Confidence score: {result.get('confidence_score', 0):.3f}")
            
            return result
        else:
            error_msg = f"API error {response.status_code}: {response.text[:200]}"
            print(f"❌ Classification failed: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except Exception as e:
        error_msg = f"Request error: {e}"
        print(f"❌ API request failed: {error_msg}")
        return {'success': False, 'error': error_msg}


def create_excel_output(pdf_path: str, extraction_result: dict, classification_result: dict, 
                       output_file: str, max_pages: int = None):
    """Create Excel file with results including page limit information."""
    
    filename = Path(pdf_path).name
    extracted_text = extraction_result.get('text', '') if extraction_result.get('success') else ''
    
    # Get categories and issues
    if classification_result.get('status') == 'success':
        categories = classification_result.get('categories', [])
        issues = classification_result.get('identified_issues', [])
        
        category_names = [cat.get('category', '') for cat in categories]
        category_confidences = [cat.get('confidence', 0.0) for cat in categories]
        issue_names = [issue.get('issue_type', '') for issue in issues]
        
        row = {
            'Filename': filename,
            'Max_Pages_Setting': max_pages if max_pages else 'All',
            'Total_Pages_In_PDF': extraction_result.get('total_pages', 0),
            'Pages_Processed': extraction_result.get('pages_processed', 0),
            'Page_Limit_Applied': extraction_result.get('page_limit_applied', False),
            'Extraction_Method': extraction_result.get('method', 'Unknown'),
            'Extraction_Time_Seconds': extraction_result.get('extraction_time', 0),
            'Characters_Extracted': len(extracted_text),
            'Extracted_Text': extracted_text,
            'Predicted_Categories_List': ', '.join(category_names),
            'Categories_Confidences_List': ', '.join([f"{conf:.3f}" for conf in category_confidences]),
            'Predicted_Issues_List': ', '.join(issue_names),
            'Overall_Confidence': classification_result.get('confidence_score', 0.0),
            'Classification_Time_Seconds': classification_result.get('processing_time', 0.0),
            'Status': 'SUCCESS'
        }
    else:
        # Handle failed classification
        error_msg = classification_result.get('error', 'Unknown error')
        row = {
            'Filename': filename,
            'Max_Pages_Setting': max_pages if max_pages else 'All',
            'Total_Pages_In_PDF': extraction_result.get('total_pages', 0),
            'Pages_Processed': extraction_result.get('pages_processed', 0),
            'Page_Limit_Applied': extraction_result.get('page_limit_applied', False),
            'Extraction_Method': extraction_result.get('method', 'Unknown'),
            'Extraction_Time_Seconds': extraction_result.get('extraction_time', 0),
            'Characters_Extracted': len(extracted_text),
            'Extracted_Text': extracted_text,
            'Predicted_Categories_List': 'CLASSIFICATION_FAILED',
            'Categories_Confidences_List': 'N/A',
            'Predicted_Issues_List': 'CLASSIFICATION_FAILED',
            'Overall_Confidence': 0.0,
            'Classification_Time_Seconds': 0.0,
            'Status': f'FAILED: {error_msg[:100]}'
        }
    
    # Create DataFrame
    df = pd.DataFrame([row])
    
    # Save to Excel
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ Excel file saved: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        return False


def process_pdf_with_page_limit(pdf_path: str, max_pages: int = None, output_file: str = None):
    """Process a single PDF with configurable page limit."""
    
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        page_suffix = f"_first{max_pages}pages" if max_pages else "_allpages"
        output_file = f"results/configurable_pdf_results_{pdf_name}{page_suffix}.xlsx"
    
    # Create results directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🧪 CONFIGURABLE PDF PROCESSING")
    print("=" * 80)
    print(f"File: {pdf_path}")
    print(f"Page limit: {max_pages if max_pages else 'No limit (all pages)'}")
    print(f"Output: {output_file}")
    print()
    
    # Step 1: Extract PDF content with page limit
    extraction_result = extract_pdf_content_with_page_limit(pdf_path, max_pages)
    if not extraction_result.get('success'):
        print(f"❌ Cannot proceed - PDF extraction failed")
        return False
    
    text = extraction_result.get('text', '')
    if not text.strip():
        print(f"❌ Cannot proceed - No text extracted from PDF")
        return False
    
    print()
    
    # Step 2: Classify the content
    classification_result = classify_with_api(text, Path(pdf_path).name)
    
    print()
    
    # Step 3: Create Excel output
    success = create_excel_output(pdf_path, extraction_result, classification_result, output_file, max_pages)
    
    # Step 4: Display summary
    if success:
        print(f"📊 PROCESSING SUMMARY:")
        print(f"   📄 PDF: {Path(pdf_path).name}")
        print(f"   📋 Pages: {extraction_result.get('pages_processed', 0)} of {extraction_result.get('total_pages', 0)} processed")
        print(f"   ⏱️  Extraction: {extraction_result.get('extraction_time', 0):.2f}s")
        print(f"   🔤 Characters: {len(text)}")
        print(f"   🎯 Classification: {'✅ Success' if classification_result.get('status') == 'success' else '❌ Failed'}")
        if classification_result.get('status') == 'success':
            categories = len(classification_result.get('categories', []))
            issues = len(classification_result.get('identified_issues', []))
            print(f"   📂 Categories: {categories}")
            print(f"   🔍 Issues: {issues}")
            print(f"   📈 Confidence: {classification_result.get('confidence_score', 0):.3f}")
        print(f"   💾 Output: {output_file}")
    
    return success


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Process PDF with configurable page limit for text extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first 2 pages only (default)
  python configurable_pdf_processor.py "data/Lot-11/filename.pdf" --max-pages 2
  
  # Process first 5 pages only
  python configurable_pdf_processor.py "data/Lot-11/filename.pdf" --max-pages 5
  
  # Process all pages
  python configurable_pdf_processor.py "data/Lot-11/filename.pdf" --max-pages 0
  
  # Process all pages (same as above)
  python configurable_pdf_processor.py "data/Lot-11/filename.pdf"
        """
    )
    
    parser.add_argument('pdf_file', help='Path to PDF file to process')
    parser.add_argument('--max-pages', type=int, default=2, 
                       help='Maximum number of pages to extract (default: 2, 0 for all pages)')
    parser.add_argument('--output', type=str, help='Output Excel file path (optional)')
    
    args = parser.parse_args()
    
    pdf_path = args.pdf_file
    max_pages = None if args.max_pages == 0 else args.max_pages
    output_file = args.output
    
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
    
    # Check if API is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Production API is not running or not responding correctly")
            print("Please start the API first: python start_production.py")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to production API at http://127.0.0.1:8000")
        print("Please start the API first: python start_production.py")
        sys.exit(1)
    
    print("✅ Production API is running")
    print()
    
    # Process the PDF
    success = process_pdf_with_page_limit(pdf_path, max_pages, output_file)
    
    if success:
        print("\n🎉 CONFIGURABLE PDF PROCESSING COMPLETED SUCCESSFULLY!")
        print("Ready for batch processing with desired page limits.")
    else:
        print("\n❌ CONFIGURABLE PDF PROCESSING FAILED!")
        print("Please check the errors above.")


if __name__ == "__main__":
    main()