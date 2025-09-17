#!/usr/bin/env python3
"""
Analyze LOT-21.xlsx ground truth file to find missing issue types in our mapper
"""

import pandas as pd
import sys
from pathlib import Path

def load_ground_truth_issues():
    """Load issue types and categories from LOT-21.xlsx ground truth file"""
    
    # LOT-21.xlsx file path
    lot21_path = "/Users/hareeshkb/work/Krishna/ccms_classification/data/Lots21-27/Lot 21 to 23/LOT-21/LOT-21.xlsx"
    
    if not Path(lot21_path).exists():
        print(f"❌ Ground truth file not found: {lot21_path}")
        return None, None
    
    try:
        # Read the Excel file with header on row 3 (index 2)
        df = pd.read_excel(lot21_path, header=2)
        
        print(f"📊 Loaded {len(df)} rows from LOT-21.xlsx")
        print(f"📊 Columns available: {list(df.columns)}")
        
        # Find columns E and F (Issues and Categories)
        # Column E = index 4, Column F = index 5
        if len(df.columns) >= 6:
            issues_col = df.columns[4]  # Column E
            category_col = df.columns[5]  # Column F
            
            print(f"📝 Column E (Issues): '{issues_col}'")
            print(f"📝 Column F (Categories): '{category_col}'")
            
            # Extract non-null values
            issues_data = df[[issues_col, category_col]].dropna()
            
            print(f"📊 Found {len(issues_data)} valid issue-category pairs")
            
            return issues_data, (issues_col, category_col)
        else:
            print(f"❌ Not enough columns found. Expected at least 6, got {len(df.columns)}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error reading LOT-21.xlsx: {e}")
        return None, None

def load_mapper_issues():
    """Load issue types from our unified mapper"""
    
    mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
    
    if not Path(mapper_path).exists():
        print(f"❌ Mapper file not found: {mapper_path}")
        return set()
    
    try:
        df = pd.read_excel(mapper_path)
        
        # Get unique issue types - try different column names
        issue_col = None
        for col in ['issue_type', 'Issue_type', 'Issue Type', 'issue', 'Issue']:
            if col in df.columns:
                issue_col = col
                break
        
        if issue_col:
            mapper_issues = set(df[issue_col].dropna().unique())
            print(f"📊 Loaded {len(mapper_issues)} issue types from mapper (column: '{issue_col}')")
            return mapper_issues
        else:
            print(f"❌ No issue type column found in mapper. Columns: {list(df.columns)}")
            return set()
            
    except Exception as e:
        print(f"❌ Error reading mapper file: {e}")
        return set()

def analyze_missing_issues():
    """Main analysis function"""
    
    print("🔍 ANALYZING LOT-21 GROUND TRUTH vs MAPPER COVERAGE")
    print("=" * 70)
    
    # Load ground truth data
    gt_data, col_names = load_ground_truth_issues()
    if gt_data is None:
        return
    
    issues_col, category_col = col_names
    
    # Load mapper data
    mapper_issues = load_mapper_issues()
    if not mapper_issues:
        return
    
    # Extract unique ground truth issues
    gt_issues = set(gt_data[issues_col].dropna().unique())
    print(f"📊 Found {len(gt_issues)} unique issue types in LOT-21 ground truth")
    
    # Find missing issues
    missing_issues = gt_issues - mapper_issues
    covered_issues = gt_issues & mapper_issues
    
    print(f"\n✅ COVERAGE ANALYSIS:")
    print(f"   📈 Issues covered by mapper: {len(covered_issues)}/{len(gt_issues)} ({len(covered_issues)/len(gt_issues)*100:.1f}%)")
    print(f"   📉 Issues missing from mapper: {len(missing_issues)}")
    
    if missing_issues:
        print(f"\n❌ MISSING ISSUE TYPES ({len(missing_issues)}):")
        print("-" * 50)
        
        # Show missing issues with their categories
        for issue in sorted(missing_issues):
            # Find categories for this issue
            categories = gt_data[gt_data[issues_col] == issue][category_col].unique()
            categories_str = ", ".join([str(cat) for cat in categories if pd.notna(cat)])
            print(f"   • '{issue}' → [{categories_str}]")
    
    if covered_issues:
        print(f"\n✅ COVERED ISSUE TYPES ({len(covered_issues)}):")
        print("-" * 50)
        
        # Sample of covered issues
        for issue in sorted(list(covered_issues)[:10]):  # Show first 10
            categories = gt_data[gt_data[issues_col] == issue][category_col].unique()
            categories_str = ", ".join([str(cat) for cat in categories if pd.notna(cat)])
            print(f"   • '{issue}' → [{categories_str}]")
        
        if len(covered_issues) > 10:
            print(f"   ... and {len(covered_issues) - 10} more")
    
    # Category analysis
    print(f"\n📊 CATEGORY DISTRIBUTION IN LOT-21:")
    print("-" * 50)
    
    category_counts = gt_data[category_col].value_counts()
    for category, count in category_counts.items():
        if pd.notna(category):
            print(f"   • {category}: {count} issues")
    
    # Detailed analysis for missing issues
    if missing_issues:
        print(f"\n🔍 DETAILED ANALYSIS OF MISSING ISSUES:")
        print("-" * 50)
        
        missing_by_category = {}
        for issue in missing_issues:
            categories = gt_data[gt_data[issues_col] == issue][category_col].unique()
            for cat in categories:
                if pd.notna(cat):
                    if cat not in missing_by_category:
                        missing_by_category[cat] = []
                    missing_by_category[cat].append(issue)
        
        for category, issues in missing_by_category.items():
            print(f"\n   📂 {category} (Missing: {len(issues)} issues):")
            for issue in sorted(issues):
                print(f"      - '{issue}'")
    
    # Export missing issues to file for easy addition
    if missing_issues:
        output_data = []
        for issue in missing_issues:
            categories = gt_data[gt_data[issues_col] == issue][category_col].unique()
            for cat in categories:
                if pd.notna(cat):
                    output_data.append({
                        'issue_type': issue,
                        'category': cat,
                        'source': 'LOT-21_ground_truth'
                    })
        
        if output_data:
            output_df = pd.DataFrame(output_data)
            output_path = "/Users/hareeshkb/work/Krishna/ccms_classification/missing_issues_from_lot21.xlsx"
            output_df.to_excel(output_path, index=False)
            print(f"\n💾 SAVED MISSING ISSUES TO: {output_path}")
            print(f"   📄 {len(output_data)} issue-category mappings ready for addition to mapper")

if __name__ == "__main__":
    analyze_missing_issues()