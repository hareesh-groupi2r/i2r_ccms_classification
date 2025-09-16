#!/usr/bin/env python3
"""
Normalize Categories in Issues-to-Category Mapping File
Reads issues_to_category_mapping_i2r.xlsx, normalizes categories, and saves as normalized version
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.category_normalizer import CategoryNormalizer

def normalize_mapping_file():
    """Normalize categories in the mapping file"""
    
    print("=" * 60)
    print("CATEGORY NORMALIZATION FOR MAPPING FILE")
    print("=" * 60)
    
    input_file = 'issues_to_category_mapping_i2r.xlsx'
    output_file = 'issues_to_category_mapping_normalized.xlsx'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    print(f"📊 Reading input file: {input_file}")
    
    # Initialize normalizer
    normalizer = CategoryNormalizer(strict_mode=False)
    print("✅ Initialized category normalizer")
    
    # Read the Excel file
    try:
        # Try to read as single sheet first
        df = pd.read_excel(input_file)
        print(f"✅ Loaded {len(df)} rows from single sheet")
        sheet_data = {'Main': df}
    except:
        # If that fails, read all sheets
        sheet_data = pd.read_excel(input_file, sheet_name=None)
        print(f"✅ Loaded {len(sheet_data)} sheets from Excel file")
    
    # Process each sheet
    normalized_sheets = {}
    normalization_stats = {
        'total_processed': 0,
        'normalized_count': 0,
        'exact_matches': 0,
        'rejected_count': 0,
        'normalization_details': []
    }
    
    for sheet_name, df in sheet_data.items():
        print(f"\n📋 Processing sheet: {sheet_name}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        
        # Find category columns (look for variations)
        category_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['category', 'categories', 'mapped_category']):
                category_columns.append(col)
        
        if not category_columns:
            print(f"   ⚠️  No category columns found, keeping sheet as-is")
            normalized_sheets[sheet_name] = df.copy()
            continue
        
        print(f"   📊 Found category columns: {category_columns}")
        
        # Create normalized version
        normalized_df = df.copy()
        
        # Process each category column
        for col in category_columns:
            print(f"\n   🔄 Normalizing column: {col}")
            
            original_values = []
            normalized_values = []
            normalization_details = []
            
            for idx, value in enumerate(df[col]):
                if pd.isna(value) or str(value).strip() == '':
                    # Keep empty values as-is
                    normalized_values.append(value)
                    continue
                
                original_category = str(value).strip()
                original_values.append(original_category)
                
                # Handle multiple categories separated by commas
                if ',' in original_category:
                    categories = [cat.strip() for cat in original_category.split(',')]
                    normalized_categories = []
                    
                    for cat in categories:
                        if cat:
                            normalized_cat, status, confidence = normalizer.normalize_category(cat)
                            normalized_categories.append(normalized_cat)
                            
                            normalization_details.append({
                                'row': idx + 1,
                                'original': cat,
                                'normalized': normalized_cat,
                                'status': status,
                                'confidence': confidence,
                                'column': col
                            })
                    
                    final_normalized = ', '.join(normalized_categories) if normalized_categories else original_category
                else:
                    # Single category
                    normalized_cat, status, confidence = normalizer.normalize_category(original_category)
                    final_normalized = normalized_cat
                    
                    normalization_details.append({
                        'row': idx + 1,
                        'original': original_category,
                        'normalized': normalized_cat,
                        'status': status,
                        'confidence': confidence,
                        'column': col
                    })
                
                normalized_values.append(final_normalized)
                normalization_stats['total_processed'] += 1
                
                if final_normalized != original_category:
                    normalization_stats['normalized_count'] += 1
                else:
                    normalization_stats['exact_matches'] += 1
            
            # Update the column with normalized values
            normalized_df[col] = normalized_values
            
            # Add a new column showing the original values for reference
            if len([d for d in normalization_details if d['normalized'] != d['original']]) > 0:
                original_col_name = f"{col}_Original"
                normalized_df[original_col_name] = df[col]
                print(f"   📋 Added original values column: {original_col_name}")
            
            # Show normalization summary for this column
            changes = len([d for d in normalization_details if d['normalized'] != d['original']])
            print(f"   📊 Normalization summary for {col}:")
            print(f"      - Total values: {len(normalization_details)}")
            print(f"      - Changed: {changes}")
            print(f"      - Unchanged: {len(normalization_details) - changes}")
            
            # Store details for overall stats
            normalization_stats['normalization_details'].extend(normalization_details)
        
        normalized_sheets[sheet_name] = normalized_df
    
    # Create summary statistics
    summary_data = [
        {'Metric': 'Total Categories Processed', 'Value': normalization_stats['total_processed'], 'Description': 'Total category values processed'},
        {'Metric': 'Categories Normalized', 'Value': normalization_stats['normalized_count'], 'Description': 'Categories that were changed during normalization'},
        {'Metric': 'Exact Matches', 'Value': normalization_stats['exact_matches'], 'Description': 'Categories that matched exactly and needed no changes'},
        {'Metric': 'Normalization Rate', 'Value': f"{(normalization_stats['normalized_count']/max(normalization_stats['total_processed'], 1))*100:.1f}%", 'Description': 'Percentage of categories that were normalized'}
    ]
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed normalization log
    if normalization_stats['normalization_details']:
        details_df = pd.DataFrame(normalization_stats['normalization_details'])
        
        # Show only the ones that were actually changed
        changed_df = details_df[details_df['original'] != details_df['normalized']].copy()
    else:
        changed_df = pd.DataFrame()
    
    # Save normalized file with multiple sheets
    print(f"\n💾 Saving normalized file: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Save all normalized sheets
        for sheet_name, df in normalized_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Add summary and details sheets
        summary_df.to_excel(writer, sheet_name='Normalization Summary', index=False)
        
        if len(changed_df) > 0:
            changed_df.to_excel(writer, sheet_name='Normalization Details', index=False)
    
    print(f"✅ Normalization completed successfully!")
    
    # Display results
    print(f"\n📊 Normalization Results:")
    print(f"   Input file: {input_file}")
    print(f"   Output file: {output_file}")
    print(f"   Total categories processed: {normalization_stats['total_processed']}")
    print(f"   Categories normalized: {normalization_stats['normalized_count']}")
    print(f"   Exact matches: {normalization_stats['exact_matches']}")
    print(f"   Normalization rate: {(normalization_stats['normalized_count']/max(normalization_stats['total_processed'], 1))*100:.1f}%")
    
    # Show examples of normalizations
    if len(changed_df) > 0:
        print(f"\n🔄 Sample Normalizations:")
        for idx, row in changed_df.head(10).iterrows():
            print(f"   '{row['original']}' → '{row['normalized']}' (confidence: {row['confidence']:.2f})")
    
    # Show category distribution after normalization
    if len(normalized_sheets) > 0:
        print(f"\n📈 Normalized Category Distribution:")
        all_categories = []
        
        for sheet_name, df in normalized_sheets.items():
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['category', 'categories']) and 'original' not in str(col).lower():
                    for value in df[col]:
                        if pd.notna(value) and str(value).strip():
                            # Handle multiple categories
                            if ',' in str(value):
                                cats = [cat.strip() for cat in str(value).split(',')]
                                all_categories.extend(cats)
                            else:
                                all_categories.append(str(value).strip())
        
        if all_categories:
            from collections import Counter
            category_counts = Counter(all_categories)
            print(f"   Total unique categories after normalization: {len(category_counts)}")
            print(f"   Top categories:")
            for cat, count in category_counts.most_common(8):
                print(f"      {cat}: {count} occurrences")
    
    print(f"\n🎉 Category normalization complete!")
    print(f"📁 Normalized file saved as: {output_file}")
    
    return output_file

if __name__ == "__main__":
    normalize_mapping_file()