#!/usr/bin/env python3
"""
Process a single PDF and save results to Excel with specified format
"""

import sys
import os
from pathlib import Path
import requests
import pandas as pd
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.pdf_extractor import PDFExtractor


def extract_pdf_content(pdf_path: str) -> dict:
    """Extract content from PDF file."""
    print(f"📄 Extracting content from: {Path(pdf_path).name}")
    
    try:
        pdf_extractor = PDFExtractor()
        text, method = pdf_extractor.extract_text(pdf_path)
        
        if text and text.strip():
            metadata = pdf_extractor.extract_metadata(pdf_path)
            print(f"✅ Extraction successful: {len(text)} characters using {method}")
            return {
                'success': True,
                'text': text.strip(),
                'metadata': metadata,
                'method': method
            }
        else:
            print(f"❌ Extraction failed: No text extracted")
            return {'success': False, 'error': f'No text extracted using method: {method}'}
            
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return {'success': False, 'error': str(e)}


def classify_with_approach(text: str, approach: str) -> dict:
    """Classify text using specific approach."""
    print(f"🔍 Classifying with approach: {approach}")
    
    payload = {
        "text": text,
        "approach": approach,
        "confidence_threshold": 0.3,
        "max_results": 10
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
            print(f"✅ Classification successful with {approach}: {processing_time:.2f}s")
            return result
        else:
            error_msg = f"API error {response.status_code}: {response.text[:200]}"
            print(f"❌ Classification failed: {error_msg}")
            return {'success': False, 'error': error_msg, 'approach': approach}
            
    except Exception as e:
        error_msg = f"Request error: {e}"
        print(f"❌ API request failed: {error_msg}")
        return {'success': False, 'error': error_msg, 'approach': approach}


def create_excel_output(pdf_path: str, extraction_result: dict, classification_results: dict, output_file: str):
    """Create Excel file with results in specified format."""
    
    filename = Path(pdf_path).name
    extracted_text = extraction_result.get('text', '') if extraction_result.get('success') else ''
    
    # Prepare data rows
    rows = []
    
    for approach, result in classification_results.items():
        if result.get('status') == 'success':
            # Get categories and confidences
            categories = result.get('categories', [])
            category_names = [cat.get('category', '') for cat in categories]
            category_confidences = [cat.get('confidence', 0.0) for cat in categories]
            
            row = {
                'Filename': filename,
                'Extracted_Text': extracted_text,
                'Predicted_Categories_List': ', '.join(category_names),
                'Categories_Confidences_List': ', '.join([f"{conf:.3f}" for conf in category_confidences]),
                'Approach_Used': approach
            }
        else:
            # Handle failed classification
            row = {
                'Filename': filename,
                'Extracted_Text': extracted_text,
                'Predicted_Categories_List': 'CLASSIFICATION_FAILED',
                'Categories_Confidences_List': 'N/A',
                'Approach_Used': approach
            }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to Excel
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✅ Excel file saved: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        return False


def process_single_pdf_to_excel(pdf_path: str, output_file: str = None):
    """Process a single PDF and save results to Excel."""
    
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        output_file = f"results/single_pdf_results_{pdf_name}.xlsx"
    
    # Create results directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🧪 PROCESSING PDF TO EXCEL")
    print("=" * 80)
    print(f"File: {pdf_path}")
    print(f"Output: {output_file}")
    print()
    
    # Step 1: Extract PDF content
    extraction_result = extract_pdf_content(pdf_path)
    if not extraction_result.get('success'):
        print(f"❌ Cannot proceed - PDF extraction failed")
        return False
    
    text = extraction_result.get('text', '')
    if not text.strip():
        print(f"❌ Cannot proceed - No text extracted from PDF")
        return False
    
    print()
    
    # Step 2: Get available approaches
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            available_approaches = []
            classifiers_loaded = health_data.get('classifiers_loaded', {})
            for approach, loaded in classifiers_loaded.items():
                if loaded:
                    available_approaches.append(approach)
        else:
            # Fallback to common approaches
            available_approaches = ['hybrid_rag', 'pure_llm']
    except:
        available_approaches = ['hybrid_rag']
    
    print(f"📋 Available approaches: {available_approaches}")
    print()
    
    # Step 3: Classify with each approach
    classification_results = {}
    
    for approach in available_approaches:
        result = classify_with_approach(text, approach)
        classification_results[approach] = result
        time.sleep(1)  # Brief pause between requests
    
    print()
    
    # Step 4: Create Excel output
    success = create_excel_output(pdf_path, extraction_result, classification_results, output_file)
    
    # Step 5: Display preview of results
    if success:
        print(f"📊 RESULTS PREVIEW:")
        df = pd.read_excel(output_file)
        print(df.to_string(index=False, max_colwidth=50))
        print()
        print(f"📈 Summary:")
        print(f"   • Rows generated: {len(df)}")
        print(f"   • Approaches tested: {len(available_approaches)}")
        print(f"   • Output file: {output_file}")
    
    return success


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python single_pdf_to_excel.py <path_to_pdf_file>")
        print("\nExample:")
        print("  python single_pdf_to_excel.py 'data/Lot-11/20201228_AE_SPK_507_Change of Scope Proposal reminder.pdf'")
        
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
    
    pdf_path = sys.argv[1]
    
    # Validate file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
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
    success = process_single_pdf_to_excel(pdf_path)
    
    if success:
        print("\n🎉 PDF TO EXCEL PROCESSING SUCCESSFUL!")
        print("Check the Excel file format before proceeding with batch processing.")
    else:
        print("\n❌ PDF TO EXCEL PROCESSING FAILED!")
        print("Please check the errors above.")


if __name__ == "__main__":
    main()