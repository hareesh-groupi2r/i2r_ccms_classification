#!/usr/bin/env python3
"""
Debug why semantic search + LLM is missing Authority Engineer and Contractor issue types
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.config_manager import ConfigManager
from classifier.hybrid_rag import HybridRAGClassifier
from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer

def debug_missing_issues():
    """Debug why we're missing contractor and authority issues"""
    
    print("🔍 DEBUGGING MISSING ISSUE TYPES")
    print("=" * 60)
    
    # Initialize components
    config_manager = ConfigManager()
    training_data_path = "data/raw/Consolidated_labeled_data.xlsx"
    
    issue_mapper = UnifiedIssueCategoryMapper(
        training_data_path=training_data_path,
        mapping_file_path="unified_issue_category_mapping.xlsx"
    )
    validator = ValidationEngine(training_data_path)
    data_analyzer = DataSufficiencyAnalyzer(training_data_path)
    
    # Initialize classifier
    classifier = HybridRAGClassifier(
        config=config_manager.config,
        issue_mapper=issue_mapper,
        validator=validator,
        data_analyzer=data_analyzer
    )
    
    print("✅ Initialized hybrid RAG classifier")
    
    # Test text that should contain authority and contractor issues
    test_text = """
    Dear Sir,
    
    UPGRADING RAJAPALYAM - SANKARANKOVIL - TIRUNELVELI (SH-41) FROM KM 14800 TO 28+000 AND 33+800 TO 82+800 
    
    As per Scope of the project, W-Beam Crash Barriers are to be provided as per Schedule-B and in accordance with the 
    Specifications and Standards mentioned in Schedule-D of the Contract agreement.
    
    But, this was not accepted by us vide our letter under reference 2 above, wherein we have stated that the all high 
    embankment locations, Minor Bridge approaches would be carried out under change of scope.
    
    In response to this instruction of Authority Engineer, vide our letter under reference 4, we proposed locations 
    in line with the Part -3 drawings and Schedule-B even though our contention was that overall crash barrier works 
    should be taken up as change of scope works.
    
    This is our contractual obligation and we are requesting your kind consideration for the same.
    
    Yours faithfully,
    For SPK and Co- KMC (JV)
    Contractor Representative
    """
    
    print("🎯 Testing with content that mentions:")
    print("   • 'Authority Engineer' (should → Authority's Obligations)")
    print("   • 'contractual obligation' (should → Contractor's Obligations)")  
    print("   • 'Contractor Representative' (should → Contractor's Obligations)")
    
    print()
    print("🔍 Running classification...")
    
    # Run classification with debug
    result = classifier.classify_text(test_text)
    
    print()
    print("📊 CLASSIFICATION RESULTS:")
    print(f"   Status: {result.get('status', 'unknown')}")
    
    if 'identified_issues' in result:
        issues = result['identified_issues']
        print(f"   🎯 Found {len(issues)} issue types:")
        for issue in issues:
            issue_type = issue.get('issue_type', 'Unknown')
            confidence = issue.get('confidence', 0)
            source = issue.get('source', 'unknown')
            print(f"     • {issue_type} (conf: {confidence:.3f}, source: {source})")
    
    if 'categories' in result:
        categories = result['categories']
        print(f"   📋 Final categories ({len(categories)}):")
        for cat in categories:
            category = cat.get('category', 'Unknown')
            confidence = cat.get('confidence', 0)
            print(f"     • {category} (conf: {confidence:.3f})")
    
    print()
    print("🚨 EXPECTED MISSING ISSUES:")
    print("   • 'Authority Engineer' → Authority's Obligations")
    print("   • 'Contractor's Obligations' related → Contractor's Obligations")
    
    # Check if these are in our training data
    print()
    print("🔍 Checking training data for these patterns...")
    
    import pandas as pd
    training_df = pd.read_excel(training_data_path)
    
    authority_samples = training_df[training_df['issue_type'].str.contains('Authority Engineer', case=False, na=False)]
    contractor_samples = training_df[training_df['issue_type'].str.contains('Contractor.*Obligation|representative', case=False, na=False)]
    
    print(f"   📊 Authority Engineer samples in training: {len(authority_samples)}")
    if len(authority_samples) > 0:
        for _, row in authority_samples.head(3).iterrows():
            print(f"     • {row['issue_type']} → {row['category']}")
    
    print(f"   📊 Contractor obligation samples in training: {len(contractor_samples)}")
    if len(contractor_samples) > 0:
        for _, row in contractor_samples.head(3).iterrows():
            print(f"     • {row['issue_type']} → {row['category']}")

if __name__ == "__main__":
    debug_missing_issues()