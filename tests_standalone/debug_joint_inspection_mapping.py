#!/usr/bin/env python3
"""
Debug script to test UnifiedIssueCategoryMapper for Joint inspection mapping
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper

def test_joint_inspection_mapping():
    """Test that Joint inspection maps to all 3 expected categories"""
    
    # Initialize the mapper with the same paths as the integrated backend
    training_data_path = "data/synthetic/combined_training_data.xlsx"
    mapping_file_path = "issue_category_mapping_diffs/unified_issue_category_mapping.xlsx"
    
    print("Initializing UnifiedIssueCategoryMapper...")
    mapper = UnifiedIssueCategoryMapper(training_data_path, mapping_file_path)
    
    print(f"Mapper initialized with {len(mapper.issue_to_categories)} issue types")
    print(f"Total categories: {len(mapper.category_frequencies)}")
    
    # Test direct category lookup for Joint inspection
    print("\n" + "="*80)
    print("Testing direct category lookup for 'Joint inspection'")
    print("="*80)
    
    categories = mapper.get_categories_for_issue("Joint inspection")
    print(f"Categories returned: {len(categories)}")
    for i, (category, confidence) in enumerate(categories):
        print(f"  {i+1}. {category} (confidence: {confidence:.3f})")
    
    # Test the full mapping flow with Joint inspection as an issue
    print("\n" + "="*80)
    print("Testing full mapping flow with Joint inspection as identified issue")
    print("="*80)
    
    # Simulate an issue as it would come from semantic search
    test_issues = [
        {
            'issue_type': 'Joint inspection',
            'confidence': 0.663,
            'evidence': 'Test evidence for joint inspection',
            'source': 'semantic_search'
        }
    ]
    
    mapped_categories = mapper.map_issues_to_categories(test_issues)
    print(f"Mapped categories: {len(mapped_categories)}")
    for i, cat_info in enumerate(mapped_categories):
        print(f"  {i+1}. Category: {cat_info['category']}")
        print(f"     Confidence: {cat_info['confidence']:.3f}")
        print(f"     Source issues: {len(cat_info['source_issues'])}")
        print(f"     Issue types: {cat_info['issue_types']}")
        print()

if __name__ == "__main__":
    test_joint_inspection_mapping()