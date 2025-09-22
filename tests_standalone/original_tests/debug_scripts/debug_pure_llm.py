#!/usr/bin/env python3
"""
Debug Pure LLM classification specifically
"""
import sys
sys.path.append('.')
import logging

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from extract_correspondence_content import CorrespondenceExtractor
from classifier.pdf_extractor import PDFExtractor
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def debug_pure_llm():
    """Debug Pure LLM classification"""
    
    print("🔍 Debugging Pure LLM Classification")
    print("=" * 50)
    
    # Initialize components
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
    
    # Test with the problematic PDF
    pdf_path = "data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf"
    
    # Extract focused content (same as in main test)
    pdf_extractor = PDFExtractor(max_pages=2)
    raw_text, method = pdf_extractor.extract_text(pdf_path)
    
    correspondence_extractor = CorrespondenceExtractor()
    focused_content = correspondence_extractor.get_focused_content(raw_text)
    
    print(f"📄 PDF: {pdf_path.split('/')[-1]}")
    print(f"📊 Focused content: {len(focused_content)} chars")
    print(f"📝 First 200 chars: {focused_content[:200]}...")
    
    # Test Pure LLM classification with debug
    print(f"\n🧠 Testing Pure LLM classification...")
    try:
        result = pure_llm_classifier.classify(focused_content)
        
        print(f"✅ Classification result keys: {result.keys()}")
        print(f"📋 Status: {result.get('status', 'unknown')}")
        print(f"🔧 LLM Provider: {result.get('llm_provider_used', 'unknown')}")
        print(f"📚 Categories: {result.get('categories', [])}")
        print(f"📊 Total categories: {len(result.get('categories', []))}")
        
        if 'validation_report' in result:
            print(f"⚠️  Validation report: {result['validation_report']}")
        
        # Show issues if available
        if 'identified_issues' in result:
            issues = result['identified_issues']
            print(f"🔍 Identified issues: {len(issues)}")
            for i, issue in enumerate(issues[:3], 1):
                print(f"  {i}. {issue.get('issue_type', 'unknown')}: {issue.get('confidence', 0)}")
        
    except Exception as e:
        print(f"❌ Pure LLM classification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pure_llm()