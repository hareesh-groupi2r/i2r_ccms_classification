#!/usr/bin/env python3
"""
Test the new UnifiedIssueCategoryMapper
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper

def test_unified_mapper():
    """Test the new unified issue mapper"""
    
    print("🧪 TESTING NEW UNIFIED ISSUE MAPPER")
    print("=" * 60)
    
    # Initialize the unified mapper
    print("📊 1. INITIALIZING UNIFIED MAPPER...")
    mapper = UnifiedIssueCategoryMapper(
        training_data_path="data/raw/Consolidated_labeled_data.xlsx",
        mapping_file_path="unified_issue_category_mapping.xlsx"
    )
    
    print(f"   ✅ Initialized mapper: {mapper}")
    print(f"   📊 Stats: {mapper.stats}")
    
    # Test the specific issues from our 2-file test
    print()
    print("🎯 2. TESTING SPECIFIC ISSUES...")
    
    test_issues = [
        {"issue_type": "Change of scope proposals clarifications", "confidence": 0.85},
        {"issue_type": "Change of scope request for additional works or works not in the scope", "confidence": 0.75},
        {"issue_type": "Rejection of COS request by Authority Engineer/Authority", "confidence": 0.68}
    ]
    
    # Test individual issue lookups first
    print("   🔍 Individual issue lookups:")
    for issue in test_issues:
        issue_type = issue["issue_type"]
        categories = mapper.get_categories_for_issue(issue_type)
        print(f"     • {issue_type}")
        print(f"       → {len(categories)} categories: {[cat[0] for cat in categories]}")
    
    # Test the full mapping process (this is what the classifier calls)
    print()
    print("🗂️  3. TESTING FULL MAPPING PROCESS...")
    
    mapped_categories = mapper.map_issues_to_categories(test_issues)
    
    print(f"   ✅ Final result: {len(mapped_categories)} categories")
    for i, cat_data in enumerate(mapped_categories):
        category = cat_data['category']
        confidence = cat_data['confidence']
        source_count = len(cat_data['source_issues'])
        print(f"     {i+1}. {category} (confidence: {confidence:.3f}, from {source_count} issues)")
    
    print()
    print("🔥 COMPARISON:")
    print(f"   New unified mapper: {len(mapped_categories)} categories")
    print(f"   Current system:     2 categories") 
    print("   This should fix the issue mapping problem!")

if __name__ == "__main__":
    test_unified_mapper()