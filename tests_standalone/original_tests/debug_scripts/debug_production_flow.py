#!/usr/bin/env python3
"""
Debug the production flow that's returning empty categories
"""
import sys
sys.path.append('.')
import logging

# Enable specific debugging for the issue
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Enable DEBUG only for the LLM classifier
pure_llm_logger = logging.getLogger('classifier.pure_llm')
pure_llm_logger.setLevel(logging.DEBUG)

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from extract_correspondence_content import CorrespondenceExtractor
from classifier.pdf_extractor import PDFExtractor
from pathlib import Path
import json

def debug_production_flow():
    """Debug the exact production flow that's returning empty results"""
    
    print("🔍 Debugging Production Flow - Pure LLM Empty Categories")
    print("=" * 60)
    
    # Initialize components exactly like in test_lot11_evaluation.py
    config_manager = ConfigManager()
    config = config_manager.get_all_config()
    
    training_data_path = Path("./data/synthetic/combined_training_data.xlsx")
    issue_mapper = IssueCategoryMapper(training_data_path)
    validator = ValidationEngine(training_data_path)
    data_analyzer = DataSufficiencyAnalyzer(training_data_path)
    
    # Initialize Pure LLM Classifier
    pure_llm_classifier = PureLLMClassifier(
        config=config['approaches']['pure_llm'],
        issue_mapper=issue_mapper,
        validator=validator,
        data_analyzer=data_analyzer
    )
    
    # Initialize extractors exactly like in production
    pdf_extractor = PDFExtractor(max_pages=2)
    correspondence_extractor = CorrespondenceExtractor()
    
    # Test with actual PDF from Lot-11 
    pdf_path = Path("./data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf")
    
    print(f"📄 Testing with: {pdf_path.name}")
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Extract and process text exactly like production
        print(f"🔄 Extracting text from PDF...")
        raw_text, extraction_method = pdf_extractor.extract_text(pdf_path)
        print(f"📝 Raw text extracted: {len(raw_text)} chars")
        print(f"🔍 Extraction method: {extraction_method}")
        
        print(f"🔄 Extracting focused correspondence content...")
        extraction_result = correspondence_extractor.extract_correspondence_content(raw_text)
        subject = extraction_result['subject']
        body = extraction_result['body']
        focused_content = f"Subject: {subject}\n\nContent: {body}"
        print(f"📧 Subject: {subject[:100]}...")
        print(f"📝 Focused content: {len(focused_content)} chars")
        print(f"📄 Body preview: {body[:200]}...")
        print(f"🔍 Extraction method: {extraction_result['extraction_method']}")
        
        # Now test Pure LLM classification with production content
        print(f"🧠 Testing Pure LLM classification with production content...")
        
        # Add more detailed logging for the classification call
        print(f"📊 About to call pure_llm_classifier.classify...")
        print(f"📄 Content length: {len(focused_content)} chars")
        
        result = pure_llm_classifier.classify(focused_content, is_file_path=False)
        
        print(f"✅ Classification completed!")
        print(f"📊 Result structure:")
        print(f"  Identified Issues: {len(result.get('identified_issues', []))}")
        print(f"  Categories: {len(result.get('categories', []))}")
        print(f"  Provider Used: {result.get('llm_provider_used', 'unknown')}")
        print(f"  Processing Time: {result.get('processing_time', 0):.2f}s")
        
        # Check each phase in detail
        print(f"\n🔍 Detailed Phase Results:")
        
        identified_issues = result.get('identified_issues', [])
        print(f"Issues identified ({len(identified_issues)}):")
        for i, issue in enumerate(identified_issues, 1):
            print(f"  {i}. {issue.get('issue_type', 'unknown')}")
            print(f"     Confidence: {issue.get('confidence', 0)}")
            print(f"     Evidence: {issue.get('evidence', 'none')[:100]}...")
        
        categories = result.get('categories', [])
        print(f"Categories mapped ({len(categories)}):")
        for i, cat in enumerate(categories, 1):
            print(f"  {i}. {cat.get('category', 'unknown')}")
            print(f"     Confidence: {cat.get('confidence', 0)}")
            print(f"     Source Issues: {[si.get('issue_type') for si in cat.get('source_issues', [])]}")
        
        # Check validation report
        validation_report = result.get('validation_report', {})
        print(f"\n📋 Validation Report:")
        print(f"  Status: {validation_report.get('validation_status', 'unknown')}")
        print(f"  Corrections: {len(validation_report.get('corrections_made', []))}")
        print(f"  Rejections: {len(validation_report.get('rejections', []))}")
        
        if not categories:
            print(f"\n❌ NO CATEGORIES FOUND! Debugging further...")
            
            # Check if issues were identified but not mapped
            if identified_issues:
                print(f"✅ Issues were identified but not mapped to categories")
                print(f"📋 Testing issue mapper directly...")
                mapped_categories = issue_mapper.map_issues_to_categories(identified_issues)
                print(f"📊 Direct mapping result: {len(mapped_categories)} categories")
                for cat in mapped_categories:
                    print(f"  - {cat.get('category', 'unknown')}: {cat.get('confidence', 0)}")
            else:
                print(f"❌ No issues were identified at all")
                print(f"📋 This suggests the LLM call itself failed or returned empty results")
        
    except Exception as e:
        print(f"❌ Production flow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_production_flow()