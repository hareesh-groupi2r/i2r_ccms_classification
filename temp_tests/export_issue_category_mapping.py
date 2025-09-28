#!/usr/bin/env python3
"""
Export Issue-to-Category Mapping to Excel
Creates a comprehensive export of the current hybrid RAG system's issue-category mappings
"""

import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.issue_mapper import IssueCategoryMapper
from classifier.category_normalizer import CategoryNormalizer
from classifier.data_sufficiency import DataSufficiencyAnalyzer


def export_issue_category_mapping():
    """Export comprehensive issue-category mapping to Excel file"""
    
    print("=" * 60)
    print("ISSUE-CATEGORY MAPPING EXPORT")
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
            break
    
    if not training_data_path:
        print("❌ No training data found. Expected locations:")
        for path in training_paths:
            print(f"   - {path}")
        return
    
    print(f"📊 Loading training data from: {training_data_path}")
    
    # Initialize components
    issue_mapper = IssueCategoryMapper(training_data_path)
    category_normalizer = CategoryNormalizer(strict_mode=False)
    data_analyzer = DataSufficiencyAnalyzer(training_data_path)
    
    # Load training data
    df = pd.read_excel(training_data_path)
    print(f"✅ Loaded {len(df)} training samples")
    
    # Create comprehensive mapping data
    mapping_data = []
    
    print("\n📋 Building comprehensive mapping...")
    
    # Get all unique issue types and their mappings
    all_issues = issue_mapper.get_all_issue_types()
    print(f"📊 Found {len(all_issues)} unique issue types")
    
    for issue_type in all_issues:
        # Get categories for this issue
        categories = issue_mapper.get_categories_for_issue(issue_type)
        
        # Get issue frequency
        issue_freq = issue_mapper.issue_frequencies.get(issue_type, 0)
        
        if categories:
            for category in categories:
                # Get category frequency
                category_freq = issue_mapper.category_frequencies.get(category, 0)
                
                # Calculate basic confidence based on frequency
                total_samples = sum(issue_mapper.issue_frequencies.values())
                confidence = min(issue_freq / max(total_samples * 0.01, 1), 1.0)
                
                mapping_data.append({
                    'Issue Type': issue_type,
                    'Category': category,
                    'Issue Frequency': issue_freq,
                    'Category Frequency': category_freq, 
                    'Mapping Confidence': round(confidence, 3),
                    'Data Sufficiency': 'Critical' if issue_freq < 5 else ('Warning' if issue_freq < 10 else 'Good'),
                    'Normalized Issue': issue_mapper.issue_normalizer.normalize_issue_type(issue_type)[0],
                    'Normalized Category': category_normalizer.normalize_category(category)[0]
                })
        else:
            # Issue with no category mappings
            mapping_data.append({
                'Issue Type': issue_type,
                'Category': 'NO_MAPPING_FOUND',
                'Issue Frequency': issue_freq,
                'Category Frequency': 0,
                'Mapping Confidence': 0.0,
                'Data Sufficiency': 'Critical' if issue_freq < 5 else ('Warning' if issue_freq < 10 else 'Good'),
                'Normalized Issue': issue_mapper.issue_normalizer.normalize_issue_type(issue_type)[0],
                'Normalized Category': 'UNMAPPED'
            })
    
    # Create DataFrames for different views
    mapping_df = pd.DataFrame(mapping_data)
    
    print(f"✅ Created {len(mapping_df)} mapping relationships")
    
    # Summary statistics
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Metric': 'Total Unique Issue Types',
        'Value': len(all_issues),
        'Description': 'Number of distinct issue types in training data'
    })
    
    summary_data.append({
        'Metric': 'Total Unique Categories', 
        'Value': len(issue_mapper.get_all_categories()),
        'Description': 'Number of distinct categories in training data'
    })
    
    summary_data.append({
        'Metric': 'Total Mapping Relationships',
        'Value': len(mapping_df),
        'Description': 'Total issue-category mapping pairs'
    })
    
    # Data sufficiency breakdown
    sufficiency_counts = mapping_df['Data Sufficiency'].value_counts()
    for sufficiency, count in sufficiency_counts.items():
        summary_data.append({
            'Metric': f'Issues - {sufficiency}',
            'Value': count,
            'Description': f'Issue types with {sufficiency.lower()} data sufficiency'
        })
    
    # Unmapped issues
    unmapped_count = len(mapping_df[mapping_df['Category'] == 'NO_MAPPING_FOUND'])
    summary_data.append({
        'Metric': 'Unmapped Issues',
        'Value': unmapped_count,
        'Description': 'Issue types with no category mappings'
    })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Category distribution
    category_stats = []
    for category in issue_mapper.get_all_categories():
        category_issues = issue_mapper.get_issues_for_category(category)
        category_freq = issue_mapper.category_frequencies.get(category, 0)
        
        category_stats.append({
            'Category': category,
            'Frequency': category_freq,
            'Unique Issues': len(category_issues),
            'Top Issues': ', '.join(list(category_issues)[:3]) if category_issues else 'None',
            'Normalized Category': category_normalizer.normalize_category(category)[0],
            'Data Sufficiency': 'Critical' if category_freq < 5 else ('Warning' if category_freq < 10 else 'Good')
        })
    
    category_df = pd.DataFrame(category_stats).sort_values('Frequency', ascending=False)
    
    # Issue type distribution  
    issue_stats = []
    for issue_type in all_issues:
        issue_categories = issue_mapper.get_categories_for_issue(issue_type)
        issue_freq = issue_mapper.issue_frequencies.get(issue_type, 0)
        
        issue_stats.append({
            'Issue Type': issue_type,
            'Frequency': issue_freq,
            'Categories Count': len(issue_categories),
            'Categories': ', '.join(list(issue_categories)) if issue_categories else 'UNMAPPED',
            'Normalized Issue': issue_mapper.issue_normalizer.normalize_issue_type(issue_type)[0],
            'Data Sufficiency': 'Critical' if issue_freq < 5 else ('Warning' if issue_freq < 10 else 'Good')
        })
    
    issue_df = pd.DataFrame(issue_stats).sort_values('Frequency', ascending=False)
    
    # Export to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'issue_category_mapping_export_{timestamp}.xlsx'
    
    print(f"\n💾 Exporting to Excel: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main mapping data
        mapping_df.to_excel(writer, sheet_name='Issue-Category Mappings', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Category distribution
        category_df.to_excel(writer, sheet_name='Category Distribution', index=False)
        
        # Issue type distribution
        issue_df.to_excel(writer, sheet_name='Issue Distribution', index=False)
        
        # Raw training data sample (first 1000 rows)
        sample_df = df.head(1000)[['issue_type', 'category', 'subject', 'body']].copy()
        sample_df.to_excel(writer, sheet_name='Training Data Sample', index=False)
    
    print(f"✅ Export completed successfully!")
    print(f"\n📊 Export Summary:")
    print(f"   File: {output_file}")
    print(f"   Total Mappings: {len(mapping_df)}")
    print(f"   Unique Issues: {len(all_issues)}")
    print(f"   Unique Categories: {len(issue_mapper.get_all_categories())}")
    print(f"   Unmapped Issues: {unmapped_count}")
    
    # Show data sufficiency breakdown
    print(f"\n📈 Data Sufficiency Breakdown:")
    for sufficiency, count in sufficiency_counts.items():
        percentage = (count / len(mapping_df)) * 100
        print(f"   {sufficiency}: {count} ({percentage:.1f}%)")
    
    # Show top categories
    print(f"\n🏆 Top 5 Categories by Frequency:")
    for idx, row in category_df.head(5).iterrows():
        print(f"   {row['Category']}: {row['Frequency']} samples ({row['Unique Issues']} issues)")
    
    # Show critical issues needing attention  
    critical_issues = issue_df[issue_df['Data Sufficiency'] == 'Critical']
    if len(critical_issues) > 0:
        print(f"\n⚠️  Critical Issues Needing More Data ({len(critical_issues)} total):")
        for idx, row in critical_issues.head(10).iterrows():
            print(f"   {row['Issue Type']}: {row['Frequency']} samples → {row['Categories']}")
    
    print(f"\n🎉 Issue-category mapping export complete!")
    print(f"📁 Open {output_file} to review the comprehensive mapping data")
    
    return output_file


if __name__ == "__main__":
    export_issue_category_mapping()