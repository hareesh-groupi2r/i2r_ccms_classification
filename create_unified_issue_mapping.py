#!/usr/bin/env python3
"""
Create Unified Issue-to-Category Mapping
Combines training data issues with explicit mapping file to create comprehensive mapping
"""

import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from classifier.category_normalizer import CategoryNormalizer
from classifier.issue_normalizer import IssueTypeNormalizer

def create_unified_mapping():
    """Create unified mapping from both training data and explicit mapping file"""
    
    print("🔧 CREATING UNIFIED ISSUE-TO-CATEGORY MAPPING")
    print("=" * 60)
    
    # Initialize normalizers
    category_normalizer = CategoryNormalizer(strict_mode=False)
    issue_normalizer = IssueTypeNormalizer()
    
    # Storage for unified mappings
    unified_mapping = defaultdict(set)
    
    print("📊 1. LOADING FROM TRAINING DATA...")
    try:
        # Load training data
        training_file = "data/raw/Consolidated_labeled_data.xlsx"
        training_df = pd.read_excel(training_file)
        print(f"   📁 Loaded {len(training_df)} training samples")
        
        training_issues = set()
        for _, row in training_df.iterrows():
            issue_type_raw = str(row['issue_type']).strip()
            categories_str = str(row['category'])
            
            # Normalize issue type
            issue_type, status, confidence = issue_normalizer.normalize_issue_type(issue_type_raw)
            if issue_type:
                training_issues.add(issue_type)
                
                # Parse and normalize categories
                categories = category_normalizer.parse_and_normalize_categories(categories_str)
                # Extract just the category names from tuples
                category_names = [cat[0] if isinstance(cat, tuple) else cat for cat in categories]
                unified_mapping[issue_type].update(category_names)
        
        print(f"   ✅ Found {len(training_issues)} unique issue types in training data")
        
    except Exception as e:
        print(f"   ❌ Error loading training data: {e}")
        training_issues = set()
    
    print()
    print("📊 2. LOADING FROM EXPLICIT MAPPING FILE...")
    try:
        # Load explicit mapping file
        mapping_file = "issues_to_category_mapping_normalized.xlsx"
        mapping_df = pd.read_excel(mapping_file)
        print(f"   📁 Loaded {len(mapping_df)} explicit mappings")
        
        mapping_issues = set()
        for _, row in mapping_df.iterrows():
            issue_type_raw = str(row['Issue_type']).strip()
            categories_raw = str(row['Category'])
            
            # Normalize issue type
            issue_type, status, confidence = issue_normalizer.normalize_issue_type(issue_type_raw)
            if issue_type and pd.notna(categories_raw):
                mapping_issues.add(issue_type)
                
                # Parse categories from comma-separated string
                if categories_raw and categories_raw != 'nan':
                    categories_list = [cat.strip() for cat in categories_raw.split(',')]
                    normalized_categories = []
                    for cat in categories_list:
                        normalized = category_normalizer.normalize_category(cat)
                        if normalized:
                            # Extract just the category name from tuple
                            cat_name = normalized[0] if isinstance(normalized, tuple) else normalized
                            normalized_categories.append(cat_name)
                    
                    # EXPLICIT MAPPING TAKES PRECEDENCE
                    # Clear training data mapping for this issue and use explicit mapping
                    unified_mapping[issue_type] = set(normalized_categories)
        
        print(f"   ✅ Found {len(mapping_issues)} unique issue types in mapping file")
        
    except Exception as e:
        print(f"   ❌ Error loading mapping file: {e}")
        mapping_issues = set()
    
    print()
    print("🔄 3. CREATING UNIFIED MAPPING...")
    
    # Convert sets to lists for final mapping
    final_mapping = {
        issue: list(categories) 
        for issue, categories in unified_mapping.items() 
        if categories  # Only include issues that have categories
    }
    
    # Analysis
    all_training_issues = training_issues
    all_mapping_issues = mapping_issues
    all_unified_issues = set(final_mapping.keys())
    
    only_in_training = all_training_issues - all_mapping_issues
    only_in_mapping = all_mapping_issues - all_training_issues
    in_both = all_training_issues & all_mapping_issues
    
    print(f"   📊 Analysis:")
    print(f"      🔸 Training data only: {len(only_in_training)} issues")
    print(f"      🔸 Mapping file only: {len(only_in_mapping)} issues") 
    print(f"      🔸 In both sources: {len(in_both)} issues")
    print(f"      🔸 Total unified: {len(all_unified_issues)} issues")
    
    print()
    print("🎯 4. TESTING SPECIFIC ISSUES...")
    
    # Test our specific issues
    test_issues = [
        "Change of scope proposals clarifications",
        "Change of scope request for additional works or works not in the scope", 
        "Rejection of Change of Scope request by Authority Engineer/Authority"
    ]
    
    for issue_type in test_issues:
        if issue_type in final_mapping:
            categories = final_mapping[issue_type]
            source = "BOTH" if issue_type in in_both else ("TRAINING" if issue_type in only_in_training else "MAPPING")
            print(f"   ✅ '{issue_type}' [{source}]")
            print(f"       → {len(categories)} categories: {categories}")
        else:
            print(f"   ❌ '{issue_type}' → NOT FOUND IN UNIFIED MAPPING")
    
    print()
    print("💾 5. SAVING UNIFIED MAPPING...")
    
    # Save to Excel for review
    output_file = "unified_issue_category_mapping.xlsx"
    mapping_list = []
    for issue, categories in final_mapping.items():
        categories_str = ", ".join(categories)
        mapping_list.append({
            "Issue_type": issue,
            "Categories": categories_str,
            "Category_count": len(categories)
        })
    
    unified_df = pd.DataFrame(mapping_list)
    unified_df = unified_df.sort_values(['Category_count', 'Issue_type'], ascending=[False, True])
    
    with pd.ExcelWriter(output_file) as writer:
        unified_df.to_excel(writer, sheet_name='Unified_Mapping', index=False)
        
        # Summary sheet
        summary_data = [
            {"Metric": "Total Issues", "Value": len(final_mapping)},
            {"Metric": "Training Data Only", "Value": len(only_in_training)},
            {"Metric": "Mapping File Only", "Value": len(only_in_mapping)},
            {"Metric": "In Both Sources", "Value": len(in_both)},
            {"Metric": "Max Categories per Issue", "Value": max(len(cats) for cats in final_mapping.values()) if final_mapping else 0},
            {"Metric": "Avg Categories per Issue", "Value": round(sum(len(cats) for cats in final_mapping.values()) / len(final_mapping), 2) if final_mapping else 0}
        ]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"   ✅ Saved unified mapping to {output_file}")
    
    return final_mapping

if __name__ == "__main__":
    create_unified_mapping()