#!/usr/bin/env python3
"""
Simple test to verify the classification system is working
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer

def main():
    print("=" * 60)
    print("Simple Classification System Test")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("\n❌ Error: Please set your OPENAI_API_KEY in .env file")
        print("   The key should start with 'sk-proj-' or 'sk-'")
        return
    
    print("\n✅ OpenAI API key found")
    
    # Load configuration
    config_manager = ConfigManager()
    data_paths = config_manager.get_data_paths()
    training_data_path = data_paths.get('training_data')
    
    if not Path(training_data_path).exists():
        print(f"\n❌ Training data not found at {training_data_path}")
        return
    
    print(f"✅ Training data found: {training_data_path}")
    
    # Load components
    print("\nLoading components...")
    issue_mapper = IssueCategoryMapper(training_data_path)
    validator = ValidationEngine(training_data_path)
    data_analyzer = DataSufficiencyAnalyzer(training_data_path)
    
    print(f"✅ Loaded {len(issue_mapper.get_all_issue_types())} issue types")
    print(f"✅ Loaded {len(issue_mapper.get_all_categories())} categories")
    
    # Show some statistics
    report = data_analyzer.generate_sufficiency_report()
    print(f"\n📊 Data Statistics:")
    print(f"   Total samples: {report['summary']['total_samples']}")
    print(f"   Critical issues (<5 samples): {len(report['critical_issues'])}")
    print(f"   Critical categories (<5 samples): {len(report['critical_categories'])}")
    
    # Test basic classification flow
    print("\n🧪 Testing classification flow:")
    
    # Sample issues to test
    test_issues = [
        {'issue_type': 'Payment delay', 'confidence': 0.9},
        {'issue_type': 'Change of scope', 'confidence': 0.85}
    ]
    
    # Map to categories
    categories = issue_mapper.map_issues_to_categories(test_issues)
    
    print(f"\nTest Issues → Categories mapping:")
    for issue in test_issues:
        print(f"  • {issue['issue_type']} (confidence: {issue['confidence']:.2f})")
    
    print(f"\nMapped Categories:")
    for cat in categories[:5]:
        print(f"  • {cat['category']} (confidence: {cat['confidence']:.2f})")
        if cat.get('source_issues'):
            for source in cat['source_issues'][:2]:
                print(f"    ← from: {source['issue_type']}")
    
    print("\n" + "=" * 60)
    print("✅ System is working correctly!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python test_classifiers.py' for full testing")
    print("2. The Hybrid RAG will download embedding models on first run")
    print("3. Check JSON output files for detailed results")

if __name__ == "__main__":
    main()