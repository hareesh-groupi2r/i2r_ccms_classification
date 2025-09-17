#!/usr/bin/env python3
"""
Update the unified mapper with new issues from EPC V3
Adds new issues and updates mapping differences while preserving existing mappings
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

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
        "contractor's obligation": "Contractor's Obligations",  # Singular form
        "contractor obligation": "Contractor's Obligations",
        
        # Change of Scope variations
        "change of scope": "Change of Scope",
        "cos": "Change of Scope",
        "change of scope ": "Change of Scope",
        "change of  scope": "Change of Scope",  # Double space
        
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
    """Load current unified mapper"""
    
    mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
    
    if not Path(mapper_path).exists():
        print(f"❌ Current mapper file not found: {mapper_path}")
        return {}, None
    
    try:
        df = pd.read_excel(mapper_path)
        
        # Find issue type column
        issue_col = None
        for col in ['issue_type', 'Issue_type', 'Issue Type', 'issue', 'Issue']:
            if col in df.columns:
                issue_col = col
                break
        
        if not issue_col:
            print(f"❌ No issue type column found in current mapper")
            return {}, None
        
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
        return current_mapping, df
        
    except Exception as e:
        print(f"❌ Error reading current mapper: {e}")
        return {}, None

def load_epc_v3_mapping():
    """Load EPC V3 mapping"""
    
    epc_v3_path = "/Users/hareeshkb/work/Krishna/ccms_classification/Issues_Rev Category_EPC_V3.xlsx"
    
    if not Path(epc_v3_path).exists():
        print(f"❌ EPC V3 file not found: {epc_v3_path}")
        return {}
    
    try:
        df = pd.read_excel(epc_v3_path, sheet_name='EPC_Issues_V3')
        
        # Column B = index 1 (Issue Type), Column C = index 2 (Category)
        issue_col = df.columns[1]  # Column B
        category_col = df.columns[2]  # Column C
        
        # Build mapping with normalized categories
        epc_v3_mapping = {}
        
        for _, row in df.iterrows():
            issue = row[issue_col]
            categories_str = row[category_col]
            
            if pd.notna(issue) and issue != "Description of the Issue":  # Skip header-like entries
                # Normalize categories
                if pd.notna(categories_str) and categories_str != "Category":  # Skip header-like entries
                    normalized_cats = CategoryNormalizer.normalize_category_list(categories_str)
                else:
                    normalized_cats = []
                
                if normalized_cats:  # Only add if we have valid categories
                    epc_v3_mapping[issue] = normalized_cats
        
        print(f"📊 Loaded {len(epc_v3_mapping)} valid issue-category mappings from EPC V3")
        return epc_v3_mapping
        
    except Exception as e:
        print(f"❌ Error reading EPC V3 file: {e}")
        return {}

def create_updated_mapper(current_mapping, epc_v3_mapping):
    """Create updated mapper by merging current and EPC V3 mappings"""
    
    print("\n🔄 CREATING UPDATED MAPPER...")
    print("-" * 50)
    
    # Start with current mapping as base
    updated_mapping = current_mapping.copy()
    
    # Track changes
    changes = {
        'new_issues': [],
        'updated_mappings': [],
        'preserved_issues': 0
    }
    
    # Process EPC V3 mappings
    for issue, epc_v3_cats in epc_v3_mapping.items():
        if issue in current_mapping:
            current_cats = set(current_mapping[issue])
            epc_v3_cats_set = set(epc_v3_cats)
            
            # Check if there are differences
            if current_cats != epc_v3_cats_set:
                # Merge categories (union of both sets)
                merged_cats = sorted(list(current_cats | epc_v3_cats_set))
                updated_mapping[issue] = merged_cats
                
                changes['updated_mappings'].append({
                    'issue': issue,
                    'current': sorted(list(current_cats)),
                    'epc_v3': epc_v3_cats,
                    'merged': merged_cats
                })
                
                print(f"🔄 Updated '{issue}':")
                print(f"   Current:  [{', '.join(sorted(list(current_cats)))}]")
                print(f"   EPC V3:   [{', '.join(epc_v3_cats)}]")
                print(f"   Merged:   [{', '.join(merged_cats)}]")
            else:
                changes['preserved_issues'] += 1
        else:
            # New issue from EPC V3
            updated_mapping[issue] = epc_v3_cats
            changes['new_issues'].append({
                'issue': issue,
                'categories': epc_v3_cats
            })
            print(f"✅ Added new issue '{issue}' → [{', '.join(epc_v3_cats)}]")
    
    print(f"\n📊 UPDATE SUMMARY:")
    print(f"   ✅ New issues added: {len(changes['new_issues'])}")
    print(f"   🔄 Mappings updated: {len(changes['updated_mappings'])}")
    print(f"   📋 Issues preserved: {changes['preserved_issues']}")
    print(f"   📈 Total issues in updated mapper: {len(updated_mapping)}")
    
    return updated_mapping, changes

def save_updated_mapper(updated_mapping, changes):
    """Save the updated mapper to Excel file"""
    
    try:
        # Create backup of current mapper first
        current_mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
        backup_path = f"/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        if Path(current_mapper_path).exists():
            import shutil
            shutil.copy2(current_mapper_path, backup_path)
            print(f"💾 Created backup: {backup_path}")
        
        # Prepare updated data
        updated_data = []
        
        for issue, categories in updated_mapping.items():
            updated_data.append({
                'Issue_type': issue,
                'Categories': ", ".join(categories),
                'Category_count': len(categories)
            })
        
        # Sort by issue type for consistency
        updated_data.sort(key=lambda x: x['Issue_type'])
        
        # Create DataFrame and save
        updated_df = pd.DataFrame(updated_data)
        updated_df.to_excel(current_mapper_path, index=False)
        
        print(f"💾 UPDATED MAPPER SAVED TO: {current_mapper_path}")
        print(f"   📄 {len(updated_data)} total issue-category mappings")
        
        # Also save a detailed change log
        change_log_data = []
        
        # Log new issues
        for change in changes['new_issues']:
            change_log_data.append({
                'Change_Type': 'NEW_ISSUE',
                'Issue_Type': change['issue'],
                'Action': 'Added from EPC V3',
                'Categories': ", ".join(change['categories']),
                'Previous_Categories': '',
                'Source': 'EPC_V3'
            })
        
        # Log updated mappings
        for change in changes['updated_mappings']:
            change_log_data.append({
                'Change_Type': 'UPDATED_MAPPING',
                'Issue_Type': change['issue'],
                'Action': 'Merged categories',
                'Categories': ", ".join(change['merged']),
                'Previous_Categories': ", ".join(change['current']),
                'Source': 'CURRENT+EPC_V3'
            })
        
        if change_log_data:
            change_log_df = pd.DataFrame(change_log_data)
            change_log_path = f"/Users/hareeshkb/work/Krishna/ccms_classification/mapper_update_changelog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            change_log_df.to_excel(change_log_path, index=False)
            print(f"📋 CHANGE LOG SAVED TO: {change_log_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving updated mapper: {e}")
        return False

def verify_update():
    """Verify the updated mapper was saved correctly"""
    
    try:
        # Read the updated mapper
        mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
        df = pd.read_excel(mapper_path)
        
        print(f"\n✅ VERIFICATION:")
        print(f"   📄 Updated mapper contains {len(df)} rows")
        print(f"   📊 Columns: {list(df.columns)}")
        
        # Show sample of new entries
        print(f"\n📋 SAMPLE OF UPDATED MAPPER (first 5 rows):")
        for i, row in df.head().iterrows():
            issue = row.get('Issue_type', 'Unknown')
            categories = row.get('Categories', 'No categories')
            print(f"   • {issue} → [{categories}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying update: {e}")
        return False

def update_integrated_backend():
    """Restart integrated backend to use updated mapper"""
    
    print(f"\n🔄 UPDATING INTEGRATED BACKEND...")
    print("-" * 50)
    
    print("ℹ️  The integrated backend will automatically reload the updated mapper")
    print("ℹ️  You may need to restart the backend server for changes to take effect")
    print("\n🎯 RECOMMENDED NEXT STEPS:")
    print("   1. Restart the integrated backend server")
    print("   2. Test with LOT-21 files to verify improved classification")
    print("   3. Run regression tests to ensure no performance degradation")

def main():
    """Main update function"""
    
    print("🔄 UNIFIED MAPPER UPDATE PROCESS")
    print("="*70)
    print("Adding new issues from EPC V3 while preserving existing mappings")
    
    # Load current mapper
    print("\n📂 Loading current mapper...")
    current_mapping, current_df = load_current_mapper()
    
    if not current_mapping:
        print("❌ Failed to load current mapper")
        return
    
    # Load EPC V3 mapper
    print("\n📂 Loading EPC V3 mapper...")
    epc_v3_mapping = load_epc_v3_mapping()
    
    if not epc_v3_mapping:
        print("❌ Failed to load EPC V3 mapper")
        return
    
    # Create updated mapper
    updated_mapping, changes = create_updated_mapper(current_mapping, epc_v3_mapping)
    
    # Show update summary
    print(f"\n✅ UPDATE SUMMARY:")
    print(f"   📈 Current issues: {len(current_mapping)}")
    print(f"   📈 New issues to add: {len(changes['new_issues'])}")
    print(f"   🔄 Mappings to update: {len(changes['updated_mappings'])}")
    print(f"   📈 Total after update: {len(updated_mapping)}")
    print(f"\n🚀 Proceeding with mapper update...")
    
    # Save updated mapper
    print(f"\n💾 Saving updated mapper...")
    if save_updated_mapper(updated_mapping, changes):
        # Verify the update
        if verify_update():
            update_integrated_backend()
            print(f"\n✅ MAPPER UPDATE COMPLETED SUCCESSFULLY!")
            print(f"📊 Added {len(changes['new_issues'])} new issues")
            print(f"🔄 Updated {len(changes['updated_mappings'])} existing mappings")
        else:
            print(f"\n⚠️ Update saved but verification failed")
    else:
        print(f"\n❌ Failed to save updated mapper")

if __name__ == "__main__":
    main()