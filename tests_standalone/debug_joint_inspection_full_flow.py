#!/usr/bin/env python3
"""
Debug script to trace exactly what happens to Joint inspection in the full classification flow
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper

def debug_joint_inspection_flow():
    """Debug the exact flow that happens in hybrid_rag.py for Joint inspection"""
    
    # Initialize the mapper with the same paths as the integrated backend
    training_data_path = "data/synthetic/combined_training_data.xlsx"
    mapping_file_path = "issue_category_mapping_diffs/unified_issue_category_mapping.xlsx"
    
    print("="*80)
    print("DEBUGGING JOINT INSPECTION MAPPING FLOW")
    print("="*80)
    
    mapper = UnifiedIssueCategoryMapper(training_data_path, mapping_file_path)
    
    # Simulate the exact scenario from the logs:
    # Phase 1 finds these 10 issues, Joint inspection is Issue #3
    similar_issues = [
        {
            'issue_type': 'Rejection of COS request by Authority Engineer/Authority',
            'confidence': 0.806,
            'evidence': 'Content: - As per Scope of the project...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Utility shifting',
            'confidence': 0.688,
            'evidence': '...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Joint inspection',  # THIS IS THE KEY ISSUE
            'confidence': 0.663,
            'evidence': '...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Change of scope proposals clarifications',
            'confidence': 0.639,
            'evidence': 'Content: - As per Scope of the project...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Submission of Design and Drawings',
            'confidence': 0.614,
            'evidence': 'After this, the Authority Engineer...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Providing Right of Way as per Schedule A',
            'confidence': 0.600,
            'evidence': '...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Modification of Appointed Date',
            'confidence': 0.590,
            'evidence': '...',
            'source': 'semantic_search'
        },
        # Add more to make 10 total
        {
            'issue_type': 'Authority Engineer',
            'confidence': 0.580,
            'evidence': '...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Design & Drawings for COS works',
            'confidence': 0.570,
            'evidence': '...',
            'source': 'semantic_search'
        },
        {
            'issue_type': 'Preliminary / preparatory works',
            'confidence': 0.560,
            'evidence': '...',
            'source': 'semantic_search'
        }
    ]
    
    print(f"Phase 1: Simulating {len(similar_issues)} issues from semantic search")
    for i, issue in enumerate(similar_issues, 1):
        print(f"  {i}. {issue['issue_type']} (conf: {issue['confidence']:.3f})")
    
    print("\n" + "="*80)
    print("PHASE 2: Testing map_issues_to_categories() with all 10 issues")
    print("="*80)
    
    # Test the exact Phase 2 mapping that happens in hybrid_rag.py line 355
    mapped_categories = mapper.map_issues_to_categories(similar_issues)
    
    print(f"Total categories mapped: {len(mapped_categories)}")
    print()
    
    for i, cat_info in enumerate(mapped_categories, 1):
        print(f"{i}. Category: {cat_info['category']}")
        print(f"   Confidence: {cat_info['confidence']:.3f}")
        print(f"   Issue types: {cat_info['issue_types']}")
        print(f"   Source issues: {len(cat_info['source_issues'])}")
        
        # Check specifically for Joint inspection
        joint_inspection_sources = [
            src for src in cat_info['source_issues'] 
            if src['issue_type'] == 'Joint inspection'
        ]
        if joint_inspection_sources:
            print(f"   🔍 JOINT INSPECTION FOUND: {len(joint_inspection_sources)} source(s)")
            for src in joint_inspection_sources:
                print(f"      - Issue: {src['issue_type']}, Confidence: {src['confidence']:.3f}")
        print()
    
    print("="*80)
    print("SPECIFIC TEST: Direct Joint inspection mapping")
    print("="*80)
    
    # Test just Joint inspection alone
    joint_only = [similar_issues[2]]  # Just the Joint inspection issue
    joint_mapped = mapper.map_issues_to_categories(joint_only)
    
    print(f"Joint inspection alone maps to {len(joint_mapped)} categories:")
    for cat in joint_mapped:
        print(f"  - {cat['category']} (confidence: {cat['confidence']:.3f})")
    
    print("\n" + "="*80)
    print("EXPECTED vs ACTUAL ANALYSIS")
    print("="*80)
    
    # What should happen
    direct_categories = mapper.get_categories_for_issue("Joint inspection")
    print(f"Expected categories from direct lookup: {len(direct_categories)}")
    for cat, conf in direct_categories:
        print(f"  - {cat} (confidence: {conf:.3f})")
    
    # Check if all expected categories are in the final result
    expected_cats = {cat for cat, _ in direct_categories}
    actual_cats = {cat['category'] for cat in mapped_categories if any(
        src['issue_type'] == 'Joint inspection' for src in cat['source_issues']
    )}
    
    print(f"\nExpected categories: {expected_cats}")
    print(f"Actual categories: {actual_cats}")
    print(f"Missing categories: {expected_cats - actual_cats}")
    
    if expected_cats - actual_cats:
        print("🚨 PROBLEM: Missing categories detected!")
        print("The 1-to-many mapping is not working correctly in the full flow.")
    else:
        print("✅ SUCCESS: All expected categories are present.")

if __name__ == "__main__":
    debug_joint_inspection_flow()