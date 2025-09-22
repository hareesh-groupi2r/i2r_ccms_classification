#!/usr/bin/env python3
"""
Fix category normalization in combined training and validation datasets
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.category_normalizer import CategoryNormalizer
from classifier.issue_normalizer import IssueTypeNormalizer


def fix_dataset_normalization(file_path: str, output_path: str = None):
    """Fix category and issue type normalization in a dataset."""
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if output_path is None:
        output_path = file_path
    
    print(f"🔧 Fixing normalization in: {file_path}")
    
    # Load data
    df = pd.read_excel(file_path)
    print(f"  Loaded {len(df)} samples")
    
    # Initialize normalizers
    category_normalizer = CategoryNormalizer()
    issue_normalizer = IssueTypeNormalizer()
    
    # Track changes
    category_changes = 0
    issue_changes = 0
    
    # Fix categories
    for idx, row in df.iterrows():
        # Normalize issue type
        issue_type_raw = str(row.get('issue_type', ''))
        issue_type_normalized, _, _ = issue_normalizer.normalize_issue_type(issue_type_raw)
        
        if issue_type_normalized != issue_type_raw:
            df.at[idx, 'issue_type'] = issue_type_normalized
            issue_changes += 1
        
        # Normalize categories
        category_raw = str(row.get('category', ''))
        normalized_cats = category_normalizer.parse_and_normalize_categories(category_raw)
        
        if normalized_cats:
            # Use the first normalized category for simplicity
            normalized_category = normalized_cats[0]
        else:
            # Fallback to 'Others' if normalization fails
            normalized_category = 'Others'
        
        if normalized_category != category_raw:
            df.at[idx, 'category'] = normalized_category
            category_changes += 1
    
    # Show statistics
    print(f"  Issue type changes: {issue_changes}")
    print(f"  Category changes: {category_changes}")
    print(f"  Final unique categories: {df['category'].nunique()}")
    print(f"  Final unique issue types: {df['issue_type'].nunique()}")
    
    # Show category distribution
    print(f"\\n📊 Category Distribution:")
    cat_counts = df['category'].value_counts()
    for category, count in cat_counts.items():
        percentage = count/len(df)*100
        print(f"  {category:<25} {count:3d} samples ({percentage:5.1f}%)")
    
    # Save fixed data
    df.to_excel(output_path, index=False)
    print(f"✅ Fixed data saved to: {output_path}")
    
    return True


def main():
    """Fix normalization in all dataset files."""
    print("=" * 60)
    print("FIXING CATEGORY NORMALIZATION IN DATASETS")
    print("=" * 60)
    
    files_to_fix = [
        './data/synthetic/combined_training_data.xlsx',
        './data/synthetic/training_set.xlsx',
        './data/synthetic/validation_set.xlsx',
    ]
    
    for file_path in files_to_fix:
        if Path(file_path).exists():
            success = fix_dataset_normalization(file_path)
            if success:
                print(f"✅ Fixed: {file_path}")
            else:
                print(f"❌ Failed: {file_path}")
            print()
        else:
            print(f"⚠️  Not found: {file_path}")
            print()
    
    # Also fix the original raw data for future use
    original_path = './data/raw/Consolidated_labeled_data.xlsx'
    fixed_path = './data/raw/Consolidated_labeled_data_normalized.xlsx'
    
    if Path(original_path).exists():
        print(f"🔧 Creating normalized version of original data...")
        success = fix_dataset_normalization(original_path, fixed_path)
        if success:
            print(f"✅ Normalized original data saved as: {fixed_path}")
    
    print("=" * 60)
    print("NORMALIZATION FIX COMPLETE")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("  1. Re-run evaluation to verify 8 categories are properly maintained")
    print("  2. All datasets should now use the standard 8 categories only")
    print("  3. Classification accuracy should improve with consistent categories")


if __name__ == "__main__":
    main()