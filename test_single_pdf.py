#!/usr/bin/env python3
"""
Test script to classify a single PDF file using the production API
"""

import sys
import os
from pathlib import Path
import requests
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.pdf_extractor import PDFExtractor


def extract_pdf_content(pdf_path: str) -> dict:
    """Extract content from PDF file."""
    print(f"📄 Extracting content from: {Path(pdf_path).name}")
    
    try:
        pdf_extractor = PDFExtractor()
        
        # extract_text returns (text, method) tuple
        text, method = pdf_extractor.extract_text(pdf_path)
        
        if text and text.strip():
            # Get metadata separately
            metadata = pdf_extractor.extract_metadata(pdf_path)
            
            print(f"✅ Extraction successful:")
            print(f"   Method: {method}")
            print(f"   Pages: {metadata.get('page_count', 0)}")
            print(f"   Characters: {len(text)}")
            print(f"   Preview: {text[:200]}...")
            
            return {
                'success': True,
                'text': text.strip(),
                'metadata': metadata,
                'method': method
            }
        else:
            print(f"❌ Extraction failed: No text extracted using method: {method}")
            return {'success': False, 'error': f'No text extracted using method: {method}'}
            
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return {'success': False, 'error': str(e)}


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
        "approach": "hybrid_rag",
        "confidence_threshold": 0.3,
        "max_results": 5
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
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {e}"
        print(f"❌ API request failed: {error_msg}")
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ Classification error: {error_msg}")
        return {'success': False, 'error': error_msg}


def display_classification_results(result: dict, filename: str):
    """Display classification results in a formatted way."""
    print(f"\n📊 CLASSIFICATION RESULTS FOR: {filename}")
    print("=" * 80)
    
    if not result or result.get('status') != 'success':
        print("❌ Classification failed or returned no results")
        return
    
    # Display identified issues
    issues = result.get('identified_issues', [])
    print(f"🔍 IDENTIFIED ISSUES ({len(issues)} found):")
    if issues:
        for i, issue in enumerate(issues, 1):
            confidence = issue.get('confidence', 0)
            source = issue.get('source', 'unknown')
            print(f"   {i}. {issue.get('issue_type', 'Unknown')}")
            print(f"      Confidence: {confidence:.3f} | Source: {source}")
    else:
        print("   No issues identified above threshold")
    
    print()
    
    # Display categories
    categories = result.get('categories', [])
    print(f"📂 IDENTIFIED CATEGORIES ({len(categories)} found):")
    if categories:
        for i, category in enumerate(categories, 1):
            confidence = category.get('confidence', 0)
            source_issues = category.get('source_issues', [])
            print(f"   {i}. {category.get('category', 'Unknown')}")
            print(f"      Confidence: {confidence:.3f}")
            if source_issues:
                print(f"      Source issues: {', '.join(source_issues[:3])}")
    else:
        print("   No categories identified above threshold")
    
    print()
    
    # Display warnings if any
    warnings = result.get('data_sufficiency_warnings', [])
    if warnings:
        print(f"⚠️  DATA SUFFICIENCY WARNINGS ({len(warnings)}):")
        for warning in warnings[:3]:
            print(f"   • {warning.get('message', 'Unknown warning')}")
        print()
    
    # Display validation report
    validation = result.get('validation_report', {})
    if validation:
        print(f"✅ VALIDATION REPORT:")
        print(f"   Hallucinations detected: {validation.get('hallucinations_detected', False)}")
        print(f"   Results valid: {validation.get('all_results_valid', True)}")
    
    print("=" * 80)


def save_results_to_file(pdf_path: str, extraction_result: dict, classification_result: dict):
    """Save results to a JSON file for record keeping."""
    results = {
        'pdf_file': str(pdf_path),
        'filename': Path(pdf_path).name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'extraction': extraction_result,
        'classification': classification_result
    }
    
    # Create results directory
    results_dir = Path('./results/pdf_classification')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    pdf_name = Path(pdf_path).stem
    output_file = results_dir / f"{pdf_name}_classification_result.json"
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None


def test_single_pdf(pdf_path: str):
    """Test classification of a single PDF file."""
    print(f"🧪 TESTING PDF CLASSIFICATION")
    print("=" * 80)
    print(f"File: {pdf_path}")
    print(f"Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
    print()
    
    # Step 1: Extract PDF content
    extraction_result = extract_pdf_content(pdf_path)
    if not extraction_result.get('success'):
        print(f"❌ Cannot proceed - PDF extraction failed")
        return False
    
    print()
    
    # Step 2: Classify the content
    text = extraction_result.get('text', '')
    if not text.strip():
        print(f"❌ Cannot proceed - No text extracted from PDF")
        return False
    
    classification_result = classify_with_api(text, Path(pdf_path).name)
    
    # Step 3: Display results
    display_classification_results(classification_result, Path(pdf_path).name)
    
    # Step 4: Save results
    results_file = save_results_to_file(pdf_path, extraction_result, classification_result)
    
    # Step 5: Summary
    success = classification_result.get('status') == 'success'
    
    print(f"\n🎯 TEST SUMMARY:")
    print(f"   PDF extraction: {'✅ Success' if extraction_result.get('success') else '❌ Failed'}")
    print(f"   Classification: {'✅ Success' if success else '❌ Failed'}")
    if success:
        issues_found = len(classification_result.get('identified_issues', []))
        categories_found = len(classification_result.get('categories', []))
        print(f"   Issues identified: {issues_found}")
        print(f"   Categories identified: {categories_found}")
        print(f"   Overall confidence: {classification_result.get('confidence_score', 0):.3f}")
    
    if results_file:
        print(f"   Results saved: ✅ {results_file}")
    
    return success


def main():
    """Main function to test a single PDF."""
    if len(sys.argv) != 2:
        print("Usage: python test_single_pdf.py <path_to_pdf_file>")
        print("\nExample:")
        print("  python test_single_pdf.py 'data/Lot-11/20201228_AE_SPK_507_Change of Scope Proposal reminder.pdf'")
        
        # Show available PDFs
        pdf_files = list(Path('data/Lot-11').glob('*.pdf'))
        if pdf_files:
            print(f"\nAvailable PDF files ({len(pdf_files)} found):")
            for i, pdf_file in enumerate(pdf_files[:10], 1):
                size_kb = pdf_file.stat().st_size / 1024
                print(f"  {i:2d}. {pdf_file.name} ({size_kb:.1f} KB)")
            if len(pdf_files) > 10:
                print(f"  ... and {len(pdf_files) - 10} more files")
        
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
        response = requests.get("http://127.0.0.1:8000/categories", timeout=5)
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
    
    # Run the test
    success = test_single_pdf(pdf_path)
    
    if success:
        print("\n🎉 PDF CLASSIFICATION TEST SUCCESSFUL!")
        print("Ready to proceed with batch processing of all PDFs.")
    else:
        print("\n❌ PDF CLASSIFICATION TEST FAILED!")
        print("Please check the errors above and resolve before batch processing.")
    
    return success


if __name__ == "__main__":
    main()