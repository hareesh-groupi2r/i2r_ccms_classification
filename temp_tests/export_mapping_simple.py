#!/usr/bin/env python3
"""
Simple Issue-to-Category Mapping Export
Direct export from training data without complex normalizations
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def export_simple_mapping():
    """Export issue-category mapping directly from training data"""
    
    print("=" * 60)
    print("SIMPLE ISSUE-CATEGORY MAPPING EXPORT")
    print("=" * 60)
    
    # Check for training data
    training_paths = [
        './data/synthetic/combined_training_data.xlsx',
        './data/raw/Consolidated_labeled_data.xlsx'
    ]
    
    training_data_path = None
    for path in training_paths:
        if Path(path).exists():
            training_data_path = path
            print(f"📊 Found training data: {path}")
            break
    
    if not training_data_path:
        print("❌ No training data found. Expected locations:")
        for path in training_paths:
            print(f"   - {path}")
        return
    
    # Load training data
    print(f"📊 Loading training data from: {training_data_path}")
    df = pd.read_excel(training_data_path)
    print(f"✅ Loaded {len(df)} training samples")
    
    # Clean and prepare data
    df['issue_type'] = df['issue_type'].fillna('Unknown Issue')
    df['category'] = df['category'].fillna('Others')
    
    print(f"\n📋 Data Overview:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Unique Issues: {df['issue_type'].nunique()}")
    print(f"   Unique Categories: {df['category'].nunique()}")
    
    # Build mapping data
    mapping_data = []
    issue_category_pairs = defaultdict(int)
    issue_frequencies = Counter(df['issue_type'])
    category_frequencies = Counter(df['category'])
    
    # Create issue-category mapping relationships
    for _, row in df.iterrows():
        issue_type = str(row['issue_type'])
        category = str(row['category'])
        
        # Handle multiple categories (split by comma)
        if ',' in category:
            categories = [cat.strip() for cat in category.split(',')]
        else:
            categories = [category]
        
        for cat in categories:
            if cat:  # Skip empty categories
                issue_category_pairs[(issue_type, cat)] += 1
    
    # Convert to mapping data
    for (issue_type, category), frequency in issue_category_pairs.items():
        issue_freq = issue_frequencies[issue_type]
        category_freq = category_frequencies[category]
        
        # Calculate mapping strength (how often this issue maps to this category)
        mapping_strength = frequency / issue_freq if issue_freq > 0 else 0
        
        mapping_data.append({
            'Issue Type': issue_type,
            'Category': category,
            'Mapping Frequency': frequency,
            'Issue Total Frequency': issue_freq,
            'Category Total Frequency': category_freq,
            'Mapping Strength': round(mapping_strength, 3),
            'Data Sufficiency': 'Critical' if issue_freq < 5 else ('Warning' if issue_freq < 10 else 'Good'),
            'Category Sufficiency': 'Critical' if category_freq < 5 else ('Warning' if category_freq < 10 else 'Good')
        })
    
    mapping_df = pd.DataFrame(mapping_data).sort_values(['Issue Type', 'Mapping Strength'], ascending=[True, False])
    
    print(f"✅ Created {len(mapping_df)} mapping relationships")
    
    # Create summary statistics
    summary_data = [
        {'Metric': 'Total Training Samples', 'Value': len(df), 'Description': 'Total number of training documents'},
        {'Metric': 'Unique Issue Types', 'Value': len(issue_frequencies), 'Description': 'Distinct issue types in training data'},
        {'Metric': 'Unique Categories', 'Value': len(category_frequencies), 'Description': 'Distinct categories in training data'},
        {'Metric': 'Total Mapping Pairs', 'Value': len(mapping_df), 'Description': 'Unique issue-category combinations'}
    ]
    
    # Data sufficiency breakdown
    critical_issues = len([freq for freq in issue_frequencies.values() if freq < 5])
    warning_issues = len([freq for freq in issue_frequencies.values() if 5 <= freq < 10])
    good_issues = len([freq for freq in issue_frequencies.values() if freq >= 10])
    
    summary_data.extend([
        {'Metric': 'Critical Issues (<5 samples)', 'Value': critical_issues, 'Description': 'Issue types needing more data'},
        {'Metric': 'Warning Issues (5-9 samples)', 'Value': warning_issues, 'Description': 'Issue types with moderate data'},
        {'Metric': 'Good Issues (10+ samples)', 'Value': good_issues, 'Description': 'Issue types with sufficient data'}
    ])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Top issues by frequency
    top_issues = pd.DataFrame([
        {'Issue Type': issue, 'Frequency': freq, 'Percentage': round(freq/len(df)*100, 1)}
        for issue, freq in issue_frequencies.most_common(20)
    ])
    
    # Top categories by frequency  
    top_categories = pd.DataFrame([
        {'Category': category, 'Frequency': freq, 'Percentage': round(freq/len(df)*100, 1)}
        for category, freq in category_frequencies.most_common(20)
    ])
    
    # Critical issues needing more data
    critical_data = [
        {
            'Issue Type': issue, 
            'Frequency': freq,
            'Categories': ', '.join(set([m['Category'] for m in mapping_data if m['Issue Type'] == issue]))
        }
        for issue, freq in issue_frequencies.items() if freq < 5
    ]
    
    critical_issues_df = pd.DataFrame(critical_data)
    if len(critical_issues_df) > 0:
        critical_issues_df = critical_issues_df.sort_values('Frequency')
    
    # Export to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'issue_category_mapping_{timestamp}.xlsx'
    
    print(f"\n💾 Exporting to Excel: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main mapping data
        mapping_df.to_excel(writer, sheet_name='Issue-Category Mappings', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Top issues
        top_issues.to_excel(writer, sheet_name='Top Issues', index=False)
        
        # Top categories
        top_categories.to_excel(writer, sheet_name='Top Categories', index=False)
        
        # Critical issues
        if len(critical_issues_df) > 0:
            critical_issues_df.to_excel(writer, sheet_name='Critical Issues', index=False)
        
        # Raw data sample
        sample_df = df[['issue_type', 'category']].head(500).copy()
        sample_df.to_excel(writer, sheet_name='Training Data Sample', index=False)
    
    print(f"✅ Export completed successfully!")
    
    # Display summary
    print(f"\n📊 Export Summary:")
    print(f"   File: {output_file}")
    print(f"   Total Mappings: {len(mapping_df)}")
    print(f"   Unique Issues: {len(issue_frequencies)}")
    print(f"   Unique Categories: {len(category_frequencies)}")
    
    print(f"\n📈 Data Sufficiency:")
    print(f"   Critical Issues (<5 samples): {critical_issues}")
    print(f"   Warning Issues (5-9 samples): {warning_issues}")
    print(f"   Good Issues (10+ samples): {good_issues}")
    
    print(f"\n🏆 Top 5 Categories:")
    for i, (category, freq) in enumerate(category_frequencies.most_common(5), 1):
        percentage = round(freq/len(df)*100, 1)
        print(f"   {i}. {category}: {freq} samples ({percentage}%)")
    
    print(f"\n🏆 Top 5 Issues:")
    for i, (issue, freq) in enumerate(issue_frequencies.most_common(5), 1):
        percentage = round(freq/len(df)*100, 1)
        print(f"   {i}. {issue}: {freq} samples ({percentage}%)")
    
    if critical_issues > 0:
        print(f"\n⚠️  Critical Issues Needing More Data:")
        for issue, freq in sorted([(issue, freq) for issue, freq in issue_frequencies.items() if freq < 5], 
                                 key=lambda x: x[1])[:10]:
            categories = set([m['Category'] for m in mapping_data if m['Issue Type'] == issue])
            print(f"   {issue}: {freq} samples → {', '.join(categories)}")
    
    print(f"\n🎉 Issue-category mapping export complete!")
    print(f"📁 File saved as: {output_file}")
    
    return output_file

if __name__ == "__main__":
    export_simple_mapping()