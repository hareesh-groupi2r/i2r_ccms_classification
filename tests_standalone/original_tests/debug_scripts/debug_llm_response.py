#!/usr/bin/env python3
"""
Debug LLM response specifically
"""
import sys
sys.path.append('.')
import logging

# Enable ALL debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from extract_correspondence_content import CorrespondenceExtractor
from classifier.pdf_extractor import PDFExtractor
from pathlib import Path

def debug_llm_response():
    """Debug the actual LLM request/response"""
    
    print("🔍 Debugging LLM Request/Response")
    print("=" * 50)
    
    # Initialize components (minimal logging for initialization)
    logger = logging.getLogger('classifier')
    logger.setLevel(logging.DEBUG)
    
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
    
    # Prepare simple test content
    test_content = """Subject: Change of Scope Proposal for toll plaza construction at Chennasamudram Village

Content: It is to inform that as per the present Scope of Works the number of Toll Lanes to be made at the Chennasamudram Toll Plaza are 24. However, based on the NHAI letter it is understood that the number of Toll Lanes to be re-designed duly considering Hybrid ETC system. We request approval for proceeding with this revised proposal for Change of Scope."""
    
    print(f"📄 Test content: {len(test_content)} chars")
    print(f"📝 Content: {test_content}")
    
    # Test just the _identify_issues method
    print(f"\n🧠 Testing _identify_issues method directly...")
    
    try:
        # Call the preprocessing first
        processed_text = pure_llm_classifier.preprocessor.normalize_document(test_content)
        print(f"📊 Processed text: {len(processed_text)} chars")
        print(f"📝 Processed: {processed_text[:200]}...")
        
        # Call _identify_issues directly
        issues, provider = pure_llm_classifier._identify_issues(processed_text)
        
        print(f"✅ LLM Provider used: {provider}")
        print(f"📚 Issues found: {len(issues)}")
        
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue.get('issue_type', 'unknown')}: confidence={issue.get('confidence', 0)}")
                print(f"     Evidence: {issue.get('evidence', 'none')[:100]}...")
        else:
            print("❌ No issues identified!")
            
    except Exception as e:
        print(f"❌ Error testing _identify_issues: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_response()