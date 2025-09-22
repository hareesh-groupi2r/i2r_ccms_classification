#!/usr/bin/env python3
"""
Enhanced single PDF processor with detailed logging for debugging and improvement
"""

import sys
import os
from pathlib import Path
import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.pdf_extractor import PDFExtractor


def setup_detailed_logging():
    """Setup comprehensive logging for classification debugging."""
    
    # Create logs directory
    logs_dir = Path('logs/classification_debug')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Configure logging
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        logs_dir / f'classification_debug_{timestamp}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    
    # Console handler for progress
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    
    # Setup logger
    logger = logging.getLogger('classification_debug')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, logs_dir / f'classification_debug_{timestamp}.log'


def extract_pdf_content(pdf_path: str, logger) -> dict:
    """Extract content from PDF file with detailed logging."""
    logger.info(f"🔍 Starting PDF extraction for: {Path(pdf_path).name}")
    
    start_time = time.time()
    
    try:
        pdf_extractor = PDFExtractor()
        
        # Log extraction attempt
        logger.debug(f"Attempting text extraction from: {pdf_path}")
        text, method = pdf_extractor.extract_text(pdf_path)
        
        extraction_time = time.time() - start_time
        
        if text and text.strip():
            metadata = pdf_extractor.extract_metadata(pdf_path)
            
            logger.info(f"✅ PDF extraction successful:")
            logger.info(f"   Method: {method}")
            logger.info(f"   Pages: {metadata.get('page_count', 0)}")
            logger.info(f"   Characters: {len(text)}")
            logger.info(f"   Extraction time: {extraction_time:.2f}s")
            logger.debug(f"   Text preview: {text[:300]}...")
            
            return {
                'success': True,
                'text': text.strip(),
                'metadata': metadata,
                'method': method,
                'extraction_time': extraction_time
            }
        else:
            logger.error(f"❌ PDF extraction failed: No text extracted using method: {method}")
            return {
                'success': False, 
                'error': f'No text extracted using method: {method}',
                'extraction_time': extraction_time
            }
            
    except Exception as e:
        extraction_time = time.time() - start_time
        logger.error(f"❌ PDF extraction error: {e}")
        logger.debug(f"   Exception details: {type(e).__name__}: {e}")
        return {
            'success': False, 
            'error': str(e),
            'extraction_time': extraction_time
        }


def classify_with_approach_detailed(text: str, approach: str, logger, filename: str) -> dict:
    """Classify text using specific approach with detailed logging."""
    logger.info(f"🚀 Starting classification with approach: {approach}")
    logger.info(f"   Document: {filename}")
    logger.info(f"   Text length: {len(text)} characters")
    
    # Truncate text if needed
    max_length = 8000
    original_length = len(text)
    if len(text) > max_length:
        text = text[:max_length] + "..."
        logger.warning(f"   Text truncated from {original_length} to {max_length} characters")
    
    payload = {
        "text": text,
        "approach": approach,
        "confidence_threshold": 0.3,
        "max_results": 10
    }
    
    # Log request details
    logger.debug(f"API request payload:")
    logger.debug(f"   approach: {approach}")
    logger.debug(f"   confidence_threshold: {payload['confidence_threshold']}")
    logger.debug(f"   max_results: {payload['max_results']}")
    logger.debug(f"   text_length: {len(payload['text'])}")
    
    try:
        start_time = time.time()
        
        logger.debug(f"Sending POST request to http://127.0.0.1:8000/classify")
        response = requests.post(
            "http://127.0.0.1:8000/classify", 
            json=payload, 
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"⏱️  API response received in {processing_time:.2f}s")
        logger.debug(f"   Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            result['total_processing_time'] = processing_time
            
            # Log detailed results
            logger.info(f"✅ Classification successful with {approach}")
            logger.info(f"   API processing time: {result.get('processing_time', 0):.2f}s")
            logger.info(f"   Total time: {processing_time:.2f}s")
            logger.info(f"   Overall confidence: {result.get('confidence_score', 0):.3f}")
            
            # Log identified issues
            issues = result.get('identified_issues', [])
            logger.info(f"   Issues identified: {len(issues)}")
            for i, issue in enumerate(issues[:5], 1):
                logger.debug(f"     {i}. {issue.get('issue_type', 'Unknown')} "
                           f"(confidence: {issue.get('confidence', 0):.3f}, "
                           f"source: {issue.get('source', 'unknown')})")
            
            # Log categories
            categories = result.get('categories', [])
            logger.info(f"   Categories identified: {len(categories)}")
            for i, cat in enumerate(categories[:5], 1):
                logger.debug(f"     {i}. {cat.get('category', 'Unknown')} "
                           f"(confidence: {cat.get('confidence', 0):.3f})")
                source_issues = cat.get('source_issues', [])
                if source_issues:
                    logger.debug(f"        Source issues: {', '.join(source_issues[:3])}")
            
            # Log warnings
            warnings = result.get('data_sufficiency_warnings', [])
            if warnings:
                logger.warning(f"   Data sufficiency warnings: {len(warnings)}")
                for warning in warnings[:3]:
                    logger.warning(f"     • {warning.get('message', 'Unknown warning')}")
            
            # Log validation results
            validation = result.get('validation_report', {})
            if validation:
                logger.debug(f"   Validation report:")
                logger.debug(f"     Hallucinations detected: {validation.get('hallucinations_detected', False)}")
                logger.debug(f"     Results valid: {validation.get('all_results_valid', True)}")
            
            return result
            
        else:
            error_msg = f"API error {response.status_code}: {response.text[:500]}"
            logger.error(f"❌ Classification failed: {error_msg}")
            logger.debug(f"   Full response: {response.text}")
            return {
                'success': False, 
                'error': error_msg, 
                'approach': approach,
                'processing_time': processing_time
            }
            
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ API request timeout: {e}")
        return {'success': False, 'error': f'Request timeout: {e}', 'approach': approach}
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API request failed: {e}")
        return {'success': False, 'error': f'Request error: {e}', 'approach': approach}
    except Exception as e:
        logger.error(f"❌ Unexpected error during classification: {e}")
        logger.debug(f"   Exception details: {type(e).__name__}: {e}")
        return {'success': False, 'error': f'Unexpected error: {e}', 'approach': approach}


def create_excel_output_detailed(pdf_path: str, extraction_result: dict, classification_results: dict, 
                                 output_file: str, logger):
    """Create Excel file with detailed results and logging."""
    logger.info(f"📊 Creating Excel output: {output_file}")
    
    filename = Path(pdf_path).name
    extracted_text = extraction_result.get('text', '') if extraction_result.get('success') else ''
    
    # Prepare data rows with enhanced information
    rows = []
    
    for approach, result in classification_results.items():
        logger.debug(f"Processing results for approach: {approach}")
        
        if result.get('status') == 'success':
            # Get categories and confidences
            categories = result.get('categories', [])
            issues = result.get('identified_issues', [])
            
            category_names = [cat.get('category', '') for cat in categories]
            category_confidences = [cat.get('confidence', 0.0) for cat in categories]
            issue_names = [issue.get('issue_type', '') for issue in issues]
            issue_confidences = [issue.get('confidence', 0.0) for issue in issues]
            
            warnings = result.get('data_sufficiency_warnings', [])
            warning_count = len(warnings)
            
            row = {
                'Filename': filename,
                'Extracted_Text': extracted_text,
                'Predicted_Categories_List': ', '.join(category_names),
                'Categories_Confidences_List': ', '.join([f"{conf:.3f}" for conf in category_confidences]),
                'Predicted_Issues_List': ', '.join(issue_names[:10]),  # Limit to top 10
                'Issues_Confidences_List': ', '.join([f"{conf:.3f}" for conf in issue_confidences[:10]]),
                'Overall_Confidence': result.get('confidence_score', 0.0),
                'Processing_Time_Seconds': result.get('processing_time', 0.0),
                'Warning_Count': warning_count,
                'Status': 'SUCCESS',
                'Approach_Used': approach
            }
            
            logger.debug(f"   {approach}: {len(category_names)} categories, {len(issue_names)} issues")
            
        else:
            # Handle failed classification
            error_msg = result.get('error', 'Unknown error')
            logger.warning(f"   {approach}: Classification failed - {error_msg}")
            
            row = {
                'Filename': filename,
                'Extracted_Text': extracted_text,
                'Predicted_Categories_List': 'CLASSIFICATION_FAILED',
                'Categories_Confidences_List': 'N/A',
                'Predicted_Issues_List': 'CLASSIFICATION_FAILED',
                'Issues_Confidences_List': 'N/A',
                'Overall_Confidence': 0.0,
                'Processing_Time_Seconds': result.get('processing_time', 0.0),
                'Warning_Count': 0,
                'Status': f'FAILED: {error_msg[:100]}',
                'Approach_Used': approach
            }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Log summary
    logger.info(f"   DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"   Columns: {', '.join(df.columns)}")
    
    # Save to Excel
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        logger.info(f"✅ Excel file saved successfully: {output_file}")
        
        # Log file statistics
        file_size = Path(output_file).stat().st_size / 1024
        logger.info(f"   File size: {file_size:.1f} KB")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save Excel file: {e}")
        logger.debug(f"   Exception details: {type(e).__name__}: {e}")
        return False


def process_single_pdf_with_enhanced_logging(pdf_path: str, output_file: str = None):
    """Process a single PDF with comprehensive logging."""
    
    # Setup logging
    logger, log_file_path = setup_detailed_logging()
    
    if output_file is None:
        pdf_name = Path(pdf_path).stem
        output_file = f"results/enhanced_single_pdf_results_{pdf_name}.xlsx"
    
    # Create results directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🔍 ENHANCED PDF CLASSIFICATION WITH DETAILED LOGGING")
    logger.info("=" * 80)
    logger.info(f"Input file: {pdf_path}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Log system information
    file_size = os.path.getsize(pdf_path) / 1024
    logger.info(f"PDF file size: {file_size:.1f} KB")
    
    # Step 1: Extract PDF content
    logger.info("\n" + "=" * 50)
    logger.info("📄 STEP 1: PDF CONTENT EXTRACTION")
    logger.info("=" * 50)
    
    extraction_result = extract_pdf_content(pdf_path, logger)
    if not extraction_result.get('success'):
        logger.error("❌ Cannot proceed - PDF extraction failed")
        return False
    
    text = extraction_result.get('text', '')
    if not text.strip():
        logger.error("❌ Cannot proceed - No text extracted from PDF")
        return False
    
    # Step 2: Get available approaches
    logger.info("\n" + "=" * 50)
    logger.info("🔧 STEP 2: SYSTEM INITIALIZATION")
    logger.info("=" * 50)
    
    try:
        logger.debug("Checking API health and available approaches...")
        response = requests.get("http://127.0.0.1:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            available_approaches = []
            classifiers_loaded = health_data.get('classifiers_loaded', {})
            
            logger.info("📋 System status:")
            logger.info(f"   API version: {health_data.get('version', 'unknown')}")
            logger.info(f"   Uptime: {health_data.get('uptime', 0):.1f} seconds")
            
            system_info = health_data.get('system_info', {})
            logger.info(f"   Training samples: {system_info.get('training_samples', 0)}")
            logger.info(f"   Issue types: {system_info.get('issue_types', 0)}")
            logger.info(f"   Categories: {system_info.get('categories', 0)}")
            
            for approach, loaded in classifiers_loaded.items():
                if loaded:
                    available_approaches.append(approach)
                    logger.info(f"   ✅ {approach} classifier: loaded")
                else:
                    logger.warning(f"   ❌ {approach} classifier: not loaded")
        else:
            logger.warning(f"API health check failed: {response.status_code}")
            available_approaches = ['hybrid_rag']
    except Exception as e:
        logger.error(f"Failed to check API health: {e}")
        available_approaches = ['hybrid_rag']
    
    logger.info(f"📊 Available approaches for testing: {available_approaches}")
    
    # Step 3: Classify with each approach
    logger.info("\n" + "=" * 50)
    logger.info("🤖 STEP 3: CLASSIFICATION WITH MULTIPLE APPROACHES")
    logger.info("=" * 50)
    
    classification_results = {}
    
    for i, approach in enumerate(available_approaches, 1):
        logger.info(f"\n--- APPROACH {i}/{len(available_approaches)}: {approach.upper()} ---")
        
        result = classify_with_approach_detailed(text, approach, logger, Path(pdf_path).name)
        classification_results[approach] = result
        
        if i < len(available_approaches):
            logger.info("⏸️  Waiting 2 seconds before next approach...")
            time.sleep(2)
    
    # Step 4: Create Excel output
    logger.info("\n" + "=" * 50)
    logger.info("📊 STEP 4: EXCEL OUTPUT GENERATION")
    logger.info("=" * 50)
    
    success = create_excel_output_detailed(
        pdf_path, extraction_result, classification_results, output_file, logger
    )
    
    # Step 5: Final summary
    logger.info("\n" + "=" * 50)
    logger.info("📈 FINAL SUMMARY")
    logger.info("=" * 50)
    
    if success:
        # Display preview of results
        try:
            df = pd.read_excel(output_file)
            logger.info("📋 Results preview:")
            for idx, row in df.iterrows():
                logger.info(f"   Row {idx + 1} - {row['Approach_Used']}:")
                logger.info(f"     Status: {row['Status']}")
                logger.info(f"     Categories: {len(str(row['Predicted_Categories_List']).split(',')) if row['Predicted_Categories_List'] != 'CLASSIFICATION_FAILED' else 0}")
                logger.info(f"     Overall confidence: {row.get('Overall_Confidence', 0):.3f}")
                logger.info(f"     Processing time: {row.get('Processing_Time_Seconds', 0):.2f}s")
        except Exception as e:
            logger.warning(f"Could not read preview: {e}")
        
        logger.info(f"✅ Processing completed successfully!")
        logger.info(f"   📄 PDF processed: {Path(pdf_path).name}")
        logger.info(f"   📊 Excel output: {output_file}")
        logger.info(f"   📝 Detailed log: {log_file_path}")
        logger.info(f"   🕒 Total approaches tested: {len(available_approaches)}")
        
    else:
        logger.error("❌ Processing failed during Excel generation")
    
    logger.info("=" * 80)
    
    return success


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_logging_single_pdf.py <path_to_pdf_file>")
        print("\nExample:")
        print("  python enhanced_logging_single_pdf.py 'data/Lot-11/20201228_AE_SPK_507_Change of Scope Proposal reminder.pdf'")
        
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
    
    print("✅ Production API is running - starting enhanced processing...")
    
    # Process the PDF with enhanced logging
    success = process_single_pdf_with_enhanced_logging(pdf_path)
    
    if success:
        print("\n🎉 ENHANCED PDF PROCESSING COMPLETED SUCCESSFULLY!")
        print("Check the logs and Excel output for detailed analysis.")
    else:
        print("\n❌ ENHANCED PDF PROCESSING FAILED!")
        print("Check the logs for detailed error information.")


if __name__ == "__main__":
    main()