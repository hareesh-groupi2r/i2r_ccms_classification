#!/usr/bin/env python3
"""
Test the Unified Issue Mapping - simulate what the classifier should be doing
"""

import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.category_normalizer import CategoryNormalizer

def simulate_proper_classification():
    """Simulate how classification should work with proper issue mapping"""
    
    print("🎭 SIMULATING PROPER CLASSIFICATION FLOW")
    print("=" * 60)
    
    # Load our unified mapping
    print("📊 1. LOADING UNIFIED MAPPING...")
    unified_df = pd.read_excel("unified_issue_category_mapping.xlsx")
    
    # Create lookup dictionary
    issue_to_categories = {}
    for _, row in unified_df.iterrows():
        issue_type = row['Issue_type']
        categories_str = row['Categories']
        categories = [cat.strip() for cat in categories_str.split(',')]
        issue_to_categories[issue_type] = categories
    
    print(f"   ✅ Loaded {len(issue_to_categories)} issue→category mappings")
    
    # Simulate the issues found by the classifier (from our test results)
    print()
    print("🔍 2. SIMULATING ISSUES FOUND BY CLASSIFIER...")
    
    found_issues = [
        {"issue_type": "Change of scope proposals clarifications", "confidence": 0.85},
        {"issue_type": "Change of scope request for additional works or works not in the scope", "confidence": 0.75},
        {"issue_type": "Rejection of COS request by Authority Engineer/Authority", "confidence": 0.68}  # Using the normalized name
    ]
    
    print("   Issues found by semantic search/LLM:")
    for issue in found_issues:
        print(f"     • {issue['issue_type']} (confidence: {issue['confidence']:.3f})")
    
    # Now apply proper issue→category mapping
    print()
    print("🗂️  3. APPLYING PROPER ISSUE→CATEGORY MAPPING...")
    
    all_categories = defaultdict(lambda: {'confidence': 0.0, 'source_issues': []})
    
    for issue in found_issues:
        issue_type = issue['issue_type']
        issue_confidence = issue['confidence']
        
        if issue_type in issue_to_categories:
            categories = issue_to_categories[issue_type]
            print(f"   ✅ {issue_type}")
            print(f"       → Maps to {len(categories)} categories: {categories}")
            
            # Each category gets the same confidence as the issue that found it
            for category in categories:
                # Take the maximum confidence if multiple issues map to same category
                if all_categories[category]['confidence'] < issue_confidence:
                    all_categories[category]['confidence'] = issue_confidence
                
                all_categories[category]['source_issues'].append({
                    'issue_type': issue_type,
                    'confidence': issue_confidence
                })
        else:
            print(f"   ❌ {issue_type} → NOT FOUND IN MAPPING")
    
    # Final result
    print()
    print("🎯 4. FINAL CLASSIFICATION RESULT...")
    print(f"   📊 Should predict {len(all_categories)} categories (instead of just 2):")
    
    sorted_categories = sorted(all_categories.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    for category, data in sorted_categories:
        confidence = data['confidence']
        source_count = len(data['source_issues'])
        print(f"     • {category} (confidence: {confidence:.3f}, from {source_count} issues)")
        for source in data['source_issues']:
            print(f"         ← {source['issue_type']}")
    
    print()
    print("🔥 PROBLEM IDENTIFIED:")
    print("   The current classification system is NOT using this comprehensive mapping!")
    print("   It's only returning 1 category per issue instead of ALL mapped categories.")
    print()
    print("✅ EXPECTED vs ACTUAL:")
    print(f"   Expected: {len(all_categories)} categories = {list(all_categories.keys())}")
    print("   Current:  2 categories = ['Payments', 'Change of Scope']")

if __name__ == "__main__":
    simulate_proper_classification()