#!/usr/bin/env python3
"""
Debug the full Pure LLM classification pipeline
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
from pathlib import Path
import json

def debug_full_pipeline():
    """Debug the full Pure LLM classification pipeline"""
    
    print("🔍 Debugging Full Pure LLM Classification Pipeline")
    print("=" * 60)
    
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
    
    # Prepare test content
    test_content = """Subject: Change of Scope Proposal for toll plaza construction at Chennasamudram Village

Content: It is to inform that as per the present Scope of Works the number of Toll Lanes to be made at the Chennasamudram Toll Plaza are 24. However, based on the NHAI letter it is understood that the number of Toll Lanes to be re-designed duly considering Hybrid ETC system. We request approval for proceeding with this revised proposal for Change of Scope."""
    
    print(f"📄 Test content: {len(test_content)} chars")
    print(f"📝 Content: {test_content}")
    
    print(f"\n🔄 Running full classification pipeline...")
    
    try:
        # Run full classification
        result = pure_llm_classifier.classify(test_content, is_file_path=False)
        
        print(f"\n✅ Classification completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # Check each phase
        print(f"\n🔍 Phase Analysis:")
        print(f"  Identified Issues: {len(result.get('identified_issues', []))}")
        for i, issue in enumerate(result.get('identified_issues', []), 1):
            print(f"    {i}. {issue.get('issue_type', 'unknown')}: confidence={issue.get('confidence', 0)}")
        
        print(f"  Categories: {len(result.get('categories', []))}")
        for i, cat in enumerate(result.get('categories', []), 1):
            print(f"    {i}. {cat.get('category', 'unknown')}: confidence={cat.get('confidence', 0)}")
        
        print(f"  Classification Path: {result.get('classification_path', 'unknown')}")
        print(f"  LLM Provider Used: {result.get('llm_provider_used', 'unknown')}")
        print(f"  Processing Time: {result.get('processing_time', 0):.2f}s")
        
        # Check validation report
        validation_report = result.get('validation_report', {})
        print(f"\n📋 Validation Report:")
        print(f"  Status: {validation_report.get('validation_status', 'unknown')}")
        print(f"  Hallucinations Detected: {validation_report.get('hallucinations_detected', False)}")
        print(f"  Corrections Made: {len(validation_report.get('corrections_made', []))}")
        print(f"  Rejections: {len(validation_report.get('rejections', []))}")
        
        # Print detailed result for analysis
        print(f"\n🔬 Full Result (JSON):")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_pipeline()