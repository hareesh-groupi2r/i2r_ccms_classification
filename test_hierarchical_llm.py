#!/usr/bin/env python3
"""
Test script for hierarchical LLM fallback system
Tests the updated pure_llm classifier with Gemini -> OpenAI -> Anthropic fallback
"""

import sys
import os
sys.path.append('.')

from classifier.config_manager import ConfigManager
from classifier.pure_llm import PureLLMClassifier
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("🧪 Testing Hierarchical LLM Fallback System")
    print("=" * 50)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        
        # Load training data
        training_data_path = "./data/synthetic/combined_training_data.xlsx"
        print(f"📊 Loading training data from: {training_data_path}")
        
        # Initialize components
        issue_mapper = IssueCategoryMapper(training_data_path)
        
        validator = ValidationEngine(training_data_path)
        
        data_analyzer = DataSufficiencyAnalyzer(training_data_path)
        
        # Get pure_llm configuration
        pure_llm_config = config_manager.get_approach_config('pure_llm')
        print(f"🔧 Pure LLM Config: {pure_llm_config}")
        
        # Initialize classifier
        print("🚀 Initializing PureLLMClassifier with hierarchical LLM support...")
        classifier = PureLLMClassifier(
            config=pure_llm_config,
            issue_mapper=issue_mapper,
            validator=validator,
            data_analyzer=data_analyzer
        )
        
        # Test document
        test_text = """
        Dear Contractor,
        
        We have identified significant delays in the project timeline for the construction of Building A. 
        The original completion date was scheduled for December 15, 2024, but current progress indicates 
        completion will not occur until February 2025.
        
        Additionally, there have been quality issues with the concrete work completed in Phase 1. 
        The concrete does not meet the specified strength requirements outlined in Section 3.2 of the contract.
        
        Please provide an updated timeline and corrective action plan to address these deficiencies.
        
        Best regards,
        Project Manager
        """
        
        print(f"📄 Testing classification with sample document...")
        print(f"Sample text preview: {test_text[:100]}...")
        
        # Classify document
        result = classifier.classify(test_text)
        
        print(f"\n✅ Classification Results:")
        print(f"🎯 LLM Provider Used: {result.get('llm_provider_used', 'unknown').upper()}")
        print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"📋 Identified Issues: {len(result.get('identified_issues', []))}")
        
        # Display identified issues
        if result.get('identified_issues'):
            print(f"\n🔍 Issues Found:")
            for i, issue in enumerate(result['identified_issues'][:3], 1):
                print(f"  {i}. {issue.get('issue_type', 'Unknown')} (confidence: {issue.get('confidence', 0):.2f})")
        
        # Display categories
        if result.get('categories'):
            print(f"\n📊 Categories Assigned:")
            for i, category in enumerate(result['categories'][:3], 1):
                print(f"  {i}. {category.get('category', 'Unknown')} (confidence: {category.get('confidence', 0):.2f})")
        
        # Display validation report
        validation_report = result.get('validation_report', {})
        if validation_report:
            print(f"\n📋 Validation Status: {validation_report.get('validation_status', 'unknown')}")
            if validation_report.get('corrections_made'):
                print(f"   Corrections made: {len(validation_report['corrections_made'])}")
        
        print(f"\n🎉 Hierarchical LLM test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)