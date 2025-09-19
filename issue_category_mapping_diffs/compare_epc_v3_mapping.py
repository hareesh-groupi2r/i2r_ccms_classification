#!/usr/bin/env python3
"""
Compare our current mapper with the new EPC_Issues_V3 mapping
Includes category normalization and detailed difference analysis
"""

import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

class CategoryNormalizer:
    """Normalize category names to standard format"""
    
    STANDARD_CATEGORIES = [
        "Authority's Obligations",
        "Contractor's Obligations", 
        "Change of Scope",
        "EoT",
        "Payments",
        "Dispute Resolution",
        "Appointed Date",
        "Completion Certificate",
        "Others"
    ]
    
    # Category normalization mapping
    CATEGORY_MAPPING = {
        # Authority variations
        "authority's obligations": "Authority's Obligations",
        "authoritys obligations": "Authority's Obligations", 
        "authority obligations": "Authority's Obligations",
        "authority's obigations": "Authority's Obligations",  # Common typo
        "authoritys obigations": "Authority's Obligations",
        "authority obigations": "Authority's Obligations",
        
        # Contractor variations
        "contractor's obligations": "Contractor's Obligations",
        "contractors obligations": "Contractor's Obligations",
        "contractor obligations": "Contractor's Obligations",
        "contactor's obligations": "Contractor's Obligations",  # Common typo
        "contactors obligations": "Contractor's Obligations",
        
        # Change of Scope variations
        "change of scope": "Change of Scope",
        "cos": "Change of Scope",
        "change of scope ": "Change of Scope",
        
        # EoT variations
        "eot": "EoT",
        "EOT": "EoT",
        "extension of time": "EoT",
        "eot ": "EoT",
        
        # Payments variations
        "payments": "Payments",
        "payment": "Payments",
        "payments ": "Payments",
        
        # Dispute Resolution variations
        "dispute resolution": "Dispute Resolution",
        "disputes": "Dispute Resolution",
        "dispute": "Dispute Resolution",
        "dispute resolution ": "Dispute Resolution",
        
        # Appointed Date variations
        "appointed date": "Appointed Date",
        "appointed date ": "Appointed Date",
        
        # Completion Certificate variations
        "completion certificate": "Completion Certificate",
        "completion certifcate": "Completion Certificate",  # Common typo
        "completion cert": "Completion Certificate",
        "completion certificate ": "Completion Certificate",
        
        # Others variations
        "others": "Others",
        "other": "Others",
        "others ": "Others"
    }
    
    @classmethod
    def normalize_category(cls, category_str):
        """Normalize a category string to standard format"""
        if pd.isna(category_str) or category_str == "":
            return None
            
        # Clean and normalize
        category_str = str(category_str).strip()
        
        # Try direct mapping first
        category_lower = category_str.lower()
        if category_lower in cls.CATEGORY_MAPPING:
            return cls.CATEGORY_MAPPING[category_lower]
        
        # Try exact match with standard categories
        if category_str in cls.STANDARD_CATEGORIES:
            return category_str
            
        # If no match found, return original but cleaned
        return category_str
    
    @classmethod
    def normalize_category_list(cls, category_str):
        """Normalize a comma-separated list of categories"""
        if pd.isna(category_str) or category_str == "":
            return []
            
        # Split by comma and normalize each
        categories = [cat.strip() for cat in str(category_str).split(',')]
        normalized = []
        
        for cat in categories:
            if cat:  # Skip empty strings
                norm_cat = cls.normalize_category(cat)
                if norm_cat and norm_cat not in normalized:
                    normalized.append(norm_cat)
        
        return sorted(normalized)

def load_current_mapper():
    """Load our current unified mapper"""
    
    mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
    
    if not Path(mapper_path).exists():
        print(f"❌ Current mapper file not found: {mapper_path}")
        return {}
    
    try:
        df = pd.read_excel(mapper_path)
        
        # Find issue type column
        issue_col = None
        for col in ['issue_type', 'Issue_type', 'Issue Type', 'issue', 'Issue']:
            if col in df.columns:
                issue_col = col
                break
        
        if not issue_col:
            print(f"❌ No issue type column found in current mapper. Columns: {list(df.columns)}")
            return {}
        
        # Build mapping with normalized categories
        current_mapping = {}
        
        for _, row in df.iterrows():
            issue = row[issue_col]
            categories_str = row.get('Categories', '') if 'Categories' in df.columns else ""
            
            if pd.notna(issue):
                # Normalize categories
                if pd.notna(categories_str):
                    normalized_cats = CategoryNormalizer.normalize_category_list(categories_str)
                else:
                    normalized_cats = []
                
                current_mapping[issue] = normalized_cats
        
        print(f"📊 Loaded {len(current_mapping)} issue-category mappings from current mapper")
        return current_mapping
        
    except Exception as e:
        print(f"❌ Error reading current mapper: {e}")
        return {}

def load_epc_v3_mapping():
    """Load the new EPC_Issues_V3 mapping"""
    
    epc_v3_path = "/Users/hareeshkb/work/Krishna/ccms_classification/Issues_Rev Category_EPC_V3.xlsx"
    
    if not Path(epc_v3_path).exists():
        print(f"❌ EPC V3 file not found: {epc_v3_path}")
        return {}
    
    try:
        # Read the specific sheet
        df = pd.read_excel(epc_v3_path, sheet_name='EPC_Issues_V3')
        
        print(f"📊 EPC V3 sheet loaded with {len(df)} rows")
        print(f"📊 Available columns: {list(df.columns)}")
        
        # Column B = index 1 (Issue Type), Column C = index 2 (Category)
        if len(df.columns) < 3:
            print(f"❌ Expected at least 3 columns in EPC V3, got {len(df.columns)}")
            return {}
        
        issue_col = df.columns[1]  # Column B
        category_col = df.columns[2]  # Column C
        
        print(f"📝 Using Column B (Issue Type): '{issue_col}'")
        print(f"📝 Using Column C (Category): '{category_col}'")
        
        # Build mapping with normalized categories
        epc_v3_mapping = {}
        
        for _, row in df.iterrows():
            issue = row[issue_col]
            categories_str = row[category_col]
            
            if pd.notna(issue):
                # Normalize categories
                if pd.notna(categories_str):
                    normalized_cats = CategoryNormalizer.normalize_category_list(categories_str)
                else:
                    normalized_cats = []
                
                epc_v3_mapping[issue] = normalized_cats
        
        print(f"📊 Loaded {len(epc_v3_mapping)} issue-category mappings from EPC V3")
        return epc_v3_mapping
        
    except Exception as e:
        print(f"❌ Error reading EPC V3 file: {e}")
        return {}

def analyze_mapping_differences(current_mapping, epc_v3_mapping):
    """Analyze differences between current and EPC V3 mappings"""
    
    print("\n" + "="*80)
    print("🔍 MAPPING COMPARISON ANALYSIS")
    print("="*80)
    
    # Get all unique issues
    current_issues = set(current_mapping.keys())
    epc_v3_issues = set(epc_v3_mapping.keys())
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   📈 Current mapper issues: {len(current_issues)}")
    print(f"   📈 EPC V3 mapper issues: {len(epc_v3_issues)}")
    print(f"   📈 Common issues: {len(current_issues & epc_v3_issues)}")
    print(f"   📈 Only in current: {len(current_issues - epc_v3_issues)}")
    print(f"   📈 Only in EPC V3: {len(epc_v3_issues - current_issues)}")
    
    # Issues only in current mapper
    only_in_current = current_issues - epc_v3_issues
    if only_in_current:
        print(f"\n❌ ISSUES ONLY IN CURRENT MAPPER ({len(only_in_current)}):")
        print("-" * 60)
        for issue in sorted(only_in_current)[:15]:  # Show first 15
            categories = current_mapping[issue]
            categories_str = ", ".join(categories) if categories else "No categories"
            print(f"   • '{issue}' → [{categories_str}]")
        if len(only_in_current) > 15:
            print(f"   ... and {len(only_in_current) - 15} more")
    
    # Issues only in EPC V3
    only_in_epc_v3 = epc_v3_issues - current_issues
    if only_in_epc_v3:
        print(f"\n✅ NEW ISSUES IN EPC V3 ({len(only_in_epc_v3)}):")
        print("-" * 60)
        for issue in sorted(only_in_epc_v3)[:15]:  # Show first 15
            categories = epc_v3_mapping[issue]
            categories_str = ", ".join(categories) if categories else "No categories"
            print(f"   • '{issue}' → [{categories_str}]")
        if len(only_in_epc_v3) > 15:
            print(f"   ... and {len(only_in_epc_v3) - 15} more")
    
    # Common issues with different mappings
    common_issues = current_issues & epc_v3_issues
    mapping_differences = {}
    
    for issue in common_issues:
        current_cats = set(current_mapping[issue])
        epc_v3_cats = set(epc_v3_mapping[issue])
        
        if current_cats != epc_v3_cats:
            mapping_differences[issue] = {
                'current': sorted(list(current_cats)),
                'epc_v3': sorted(list(epc_v3_cats)),
                'missing_in_current': sorted(list(epc_v3_cats - current_cats)),
                'extra_in_current': sorted(list(current_cats - epc_v3_cats))
            }
    
    if mapping_differences:
        print(f"\n🔄 ISSUES WITH DIFFERENT CATEGORY MAPPINGS ({len(mapping_differences)}):")
        print("-" * 60)
        
        for issue, diff in list(mapping_differences.items())[:15]:  # Show first 15
            print(f"\n   🔄 '{issue}':")
            print(f"      Current:  [{', '.join(diff['current'])}]")
            print(f"      EPC V3:   [{', '.join(diff['epc_v3'])}]")
            if diff['missing_in_current']:
                print(f"      Missing:  [{', '.join(diff['missing_in_current'])}]")
            if diff['extra_in_current']:
                print(f"      Extra:    [{', '.join(diff['extra_in_current'])}]")
        
        if len(mapping_differences) > 15:
            print(f"\n   ... and {len(mapping_differences) - 15} more differences")
    
    # Category usage analysis
    print(f"\n📊 CATEGORY USAGE ANALYSIS:")
    print("-" * 60)
    
    # Count category usage in both mappings
    current_cat_usage = defaultdict(int)
    epc_v3_cat_usage = defaultdict(int)
    
    for categories in current_mapping.values():
        for cat in categories:
            current_cat_usage[cat] += 1
    
    for categories in epc_v3_mapping.values():
        for cat in categories:
            epc_v3_cat_usage[cat] += 1
    
    # All categories used
    all_categories = set(current_cat_usage.keys()) | set(epc_v3_cat_usage.keys())
    
    print(f"   📂 Category usage comparison:")
    for cat in sorted(all_categories):
        current_count = current_cat_usage.get(cat, 0)
        epc_v3_count = epc_v3_cat_usage.get(cat, 0)
        diff = epc_v3_count - current_count
        diff_str = f"({diff:+d})" if diff != 0 else ""
        print(f"      {cat}: Current={current_count}, EPC V3={epc_v3_count} {diff_str}")
    
    return {
        'only_in_current': only_in_current,
        'only_in_epc_v3': only_in_epc_v3,
        'mapping_differences': mapping_differences,
        'current_cat_usage': dict(current_cat_usage),
        'epc_v3_cat_usage': dict(epc_v3_cat_usage)
    }

def export_comparison_results(current_mapping, epc_v3_mapping, analysis_results):
    """Export detailed comparison results to Excel files"""
    
    try:
        # Export 1: Side-by-side comparison of all issues
        all_issues = set(current_mapping.keys()) | set(epc_v3_mapping.keys())
        
        comparison_data = []
        for issue in sorted(all_issues):
            current_cats = current_mapping.get(issue, [])
            epc_v3_cats = epc_v3_mapping.get(issue, [])
            
            status = "SAME"
            if issue not in current_mapping:
                status = "NEW_IN_EPC_V3"
            elif issue not in epc_v3_mapping:
                status = "ONLY_IN_CURRENT"
            elif set(current_cats) != set(epc_v3_cats):
                status = "DIFFERENT"
            
            comparison_data.append({
                'Issue_Type': issue,
                'Status': status,
                'Current_Categories': ", ".join(current_cats),
                'EPC_V3_Categories': ", ".join(epc_v3_cats),
                'Current_Count': len(current_cats),
                'EPC_V3_Count': len(epc_v3_cats),
                'Missing_In_Current': ", ".join(sorted(list(set(epc_v3_cats) - set(current_cats)))),
                'Extra_In_Current': ", ".join(sorted(list(set(current_cats) - set(epc_v3_cats))))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = "/Users/hareeshkb/work/Krishna/ccms_classification/mapper_comparison_detailed.xlsx"
        comparison_df.to_excel(comparison_path, index=False)
        print(f"\n💾 EXPORTED DETAILED COMPARISON TO: {comparison_path}")
        
        # Export 2: Summary of changes needed
        changes_needed = []
        
        # New issues to add
        for issue in analysis_results['only_in_epc_v3']:
            categories = epc_v3_mapping[issue]
            changes_needed.append({
                'Change_Type': 'ADD_NEW_ISSUE',
                'Issue_Type': issue,
                'Action': 'Add to mapper',
                'New_Categories': ", ".join(categories),
                'Current_Categories': '',
                'Priority': 'HIGH' if categories else 'MEDIUM'
            })
        
        # Issues to update
        for issue, diff in analysis_results['mapping_differences'].items():
            if diff['missing_in_current']:
                changes_needed.append({
                    'Change_Type': 'UPDATE_CATEGORIES',
                    'Issue_Type': issue,
                    'Action': 'Add missing categories',
                    'New_Categories': ", ".join(diff['epc_v3']),
                    'Current_Categories': ", ".join(diff['current']),
                    'Priority': 'HIGH'
                })
        
        # Issues to remove (optional)
        for issue in list(analysis_results['only_in_current'])[:20]:  # Limit to 20
            categories = current_mapping[issue]
            changes_needed.append({
                'Change_Type': 'REVIEW_REMOVAL',
                'Issue_Type': issue,
                'Action': 'Consider removing (not in EPC V3)',
                'New_Categories': '',
                'Current_Categories': ", ".join(categories),
                'Priority': 'LOW'
            })
        
        if changes_needed:
            changes_df = pd.DataFrame(changes_needed)
            changes_path = "/Users/hareeshkb/work/Krishna/ccms_classification/mapper_changes_needed.xlsx"
            changes_df.to_excel(changes_path, index=False)
            print(f"💾 EXPORTED CHANGES NEEDED TO: {changes_path}")
            print(f"   📄 {len(changes_needed)} recommended changes")
        
        # Export 3: Updated mapper template
        updated_mapping_data = []
        
        # Start with EPC V3 as the base (most recent)
        for issue, categories in epc_v3_mapping.items():
            updated_mapping_data.append({
                'Issue_type': issue,
                'Categories': ", ".join(categories),
                'Category_count': len(categories),
                'Source': 'EPC_V3'
            })
        
        # Add issues only in current mapper (marked for review)
        for issue in analysis_results['only_in_current']:
            categories = current_mapping[issue]
            updated_mapping_data.append({
                'Issue_type': issue,
                'Categories': ", ".join(categories),
                'Category_count': len(categories),
                'Source': 'CURRENT_ONLY'
            })
        
        updated_df = pd.DataFrame(updated_mapping_data)
        updated_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_mapper_updated_template.xlsx"
        updated_df.to_excel(updated_path, index=False)
        print(f"💾 EXPORTED UPDATED MAPPER TEMPLATE TO: {updated_path}")
        
    except Exception as e:
        print(f"\n⚠️ Error exporting comparison results: {e}")

def main():
    """Main comparison function"""
    
    print("🔍 EPC V3 MAPPING COMPARISON ANALYSIS")
    print("="*70)
    
    # Load current mapper
    print("\n📂 Loading current mapper...")
    current_mapping = load_current_mapper()
    
    if not current_mapping:
        print("❌ Failed to load current mapper")
        return
    
    # Load EPC V3 mapper
    print("\n📂 Loading EPC V3 mapper...")
    epc_v3_mapping = load_epc_v3_mapping()
    
    if not epc_v3_mapping:
        print("❌ Failed to load EPC V3 mapper")
        return
    
    # Analyze differences
    analysis_results = analyze_mapping_differences(current_mapping, epc_v3_mapping)
    
    # Export results
    export_comparison_results(current_mapping, epc_v3_mapping, analysis_results)
    
    print(f"\n✅ COMPARISON ANALYSIS COMPLETE!")
    print(f"📊 Current mapper: {len(current_mapping)} issues")
    print(f"📊 EPC V3 mapper: {len(epc_v3_mapping)} issues")
    print(f"📊 New issues in EPC V3: {len(analysis_results['only_in_epc_v3'])}")
    print(f"📊 Mapping differences: {len(analysis_results['mapping_differences'])}")

if __name__ == "__main__":
    main()