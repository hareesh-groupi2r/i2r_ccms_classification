#!/usr/bin/env python3
"""
Test what the actual IssueCategoryMapper is loading and compare with our expected mapping
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.issue_mapper import IssueCategoryMapper

def test_actual_vs_expected():
    """Test current vs expected mapping"""
    
    print("🔍 TESTING CURRENT vs EXPECTED ISSUE MAPPING")
    print("=" * 60)
    
    # 1. Test what the current system loads
    print("📊 1. CURRENT SYSTEM MAPPING (from training data):")
    try:
        current_mapper = IssueCategoryMapper("data/raw/Consolidated_labeled_data.xlsx")
        
        # Check our test issue types
        test_issues = [
            "Change of scope proposals clarifications",
            "Change of scope request for additional works or works not in the scope", 
            "Rejection of Change of Scope request by Authority Engineer/Authority"
        ]
        
        print(f"   🔍 Testing {len(test_issues)} issue types...")
        for issue_type in test_issues:
            categories = current_mapper.get_categories_for_issue(issue_type)
            if categories:
                category_names = [cat[0] for cat in categories]  # Extract just the category names
                print(f"   ✅ '{issue_type}' → {len(category_names)} categories: {category_names}")
            else:
                print(f"   ❌ '{issue_type}' → NO MAPPING FOUND")
                
    except Exception as e:
        print(f"   ❌ Error loading current mapper: {e}")
    
    print()
    print("📊 2. EXPECTED MAPPING (from normalized file):")
    
    # 2. Test what the normalized Excel file contains
    normalized_file = "issues_to_category_mapping_normalized.xlsx"
    try:
        df = pd.read_excel(normalized_file)
        print(f"   📁 Loaded {len(df)} mappings from {normalized_file}")
        print(f"   📋 Columns: {df.columns.tolist()}")
        
        for issue_type in test_issues:
            matching_rows = df[df['Issue_type'].str.contains(issue_type, case=False, na=False)]
            if len(matching_rows) > 0:
                categories_raw = matching_rows.iloc[0]['Category']
                if pd.notna(categories_raw):
                    categories = [cat.strip() for cat in str(categories_raw).split(',')]
                    print(f"   ✅ '{issue_type}' → {len(categories)} categories: {categories}")
                else:
                    print(f"   ❌ '{issue_type}' → NULL CATEGORIES")
            else:
                print(f"   ❌ '{issue_type}' → NOT FOUND IN FILE")
                
    except Exception as e:
        print(f"   ❌ Error loading normalized file: {e}")
    
    print()
    print("🚨 ISSUE ANALYSIS:")
    print("   The IssueCategoryMapper is loading from training data, NOT the dedicated mapping file!")
    print("   This means it's building mappings from individual training samples,")
    print("   not using the explicit issue→category mappings you've defined.")

if __name__ == "__main__":
    test_actual_vs_expected()