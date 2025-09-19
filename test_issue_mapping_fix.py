#!/usr/bin/env python3
"""
Quick test to demonstrate and fix the issue-to-category mapping problem
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.issue_mapper import IssueCategoryMapper
from classifier.category_normalizer import CategoryNormalizer
import pandas as pd

def test_issue_mapping():
    """Test issue mapping functionality"""
    print("🔍 Testing Issue-to-Category Mapping")
    print("=" * 50)
    
    # Initialize components
    category_normalizer = CategoryNormalizer(strict_mode=False)
    
    # Use the Excel file instead of CSV which has parsing issues
    mapping_file = "/Users/hareeshkb/work/Krishna/ccms_classification/issues_to_category_mapping_normalized.xlsx"
    
    print(f"📂 Loading mapping from: {mapping_file}")
    
    # First, create a simple manual mapper to demonstrate the problem
    manual_mapping = {}
    
    # Read the Excel file to understand the format
    try:
        df = pd.read_excel(mapping_file)
        print(f"📊 CSV columns: {df.columns.tolist()}")
        print(f"📊 First few rows:")
        print(df.head())
        
        # Check specific issue types we found
        test_issues = [
            "Change of scope proposals clarifications",
            "Change of scope request for additional works or works not in the scope", 
            "Rejection of Change of Scope request by Authority Engineer/Authority"
        ]
        
        print(f"\n🎯 Testing specific issue types:")
        
        for issue_type in test_issues:
            print(f"\n🔍 Issue Type: '{issue_type}'")
            
            # Find the row with this issue type
            matching_rows = df[df.iloc[:, 0].str.contains(issue_type, case=False, na=False)]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                categories_raw = row.iloc[1] if len(row) > 1 else ""
                print(f"   📋 Raw mapping: '{categories_raw}'")
                
                # Parse categories (they seem to be comma-separated)
                if categories_raw and isinstance(categories_raw, str):
                    categories_list = [cat.strip() for cat in categories_raw.split(',')]
                    print(f"   📋 Parsed categories: {categories_list}")
                    
                    # Normalize each category
                    normalized_categories = []
                    for cat in categories_list:
                        normalized = category_normalizer.normalize_category(cat)
                        normalized_categories.append(normalized)
                    
                    print(f"   ✅ Normalized categories: {normalized_categories}")
                    print(f"   📊 Total categories: {len(normalized_categories)}")
                    
                    # Store in manual mapping for later use
                    manual_mapping[issue_type] = normalized_categories
                else:
                    print(f"   ❌ No categories found")
            else:
                print(f"   ❌ Issue type not found in mapping")
                
        print(f"\n🧪 Now test the issue→category mapping:")
        
        # Build manual mapping from the CSV
        for issue_type in test_issues:
            if issue_type in manual_mapping:
                categories_list = manual_mapping[issue_type]
                print(f"   🔍 '{issue_type}' → {categories_list}")
                print(f"       📊 Count: {len(categories_list)} categories")
            else:
                print(f"   ❌ '{issue_type}' not found in mapping")
        
        # Now test what happens when we simulate the classification flow
        print(f"\n🎭 Simulating the classification flow:")
        print("   1. Issues found by semantic search/LLM:")
        simulated_issues = [
            {"issue_type": "Change of scope proposals clarifications", "confidence": 0.85},
            {"issue_type": "Change of scope request for additional works or works not in the scope", "confidence": 0.75},
            {"issue_type": "Rejection of Change of Scope request by Authority Engineer/Authority", "confidence": 0.68}
        ]
        
        for issue in simulated_issues:
            issue_type = issue["issue_type"]
            confidence = issue["confidence"]
            if issue_type in manual_mapping:
                categories = manual_mapping[issue_type]
                print(f"   📋 Issue: '{issue_type}' (conf: {confidence:.3f})")
                print(f"       → Should map to {len(categories)} categories: {categories}")
            else:
                print(f"   ❌ Issue: '{issue_type}' not found in mapping")
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

if __name__ == "__main__":
    test_issue_mapping()