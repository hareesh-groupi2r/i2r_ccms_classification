#!/usr/bin/env python3
"""
Test script for Category Normalizer
Validates category normalization and data quality checks
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.category_normalizer import CategoryNormalizer


def test_normalizer():
    """Test the category normalizer with various inputs."""
    print("=" * 80)
    print("Testing Category Normalizer")
    print("=" * 80)
    
    # Initialize normalizer
    normalizer = CategoryNormalizer(strict_mode=False)
    
    # Test cases with various category variations
    test_cases = [
        # Standard categories
        ("EoT", "exact match"),
        ("Payments", "exact match"),
        
        # Common variations
        ("eot", "should normalize to EoT"),
        ("EOT", "should normalize to EoT"),
        ("payment", "should normalize to Payments"),
        ("payments", "should normalize to Payments"),
        ("cos", "should normalize to Change of Scope"),
        ("COS", "should normalize to Change of Scope"),
        ("change of scope", "should normalize to Change of Scope"),
        ("Change of scope", "should normalize to Change of Scope"),
        
        # Contractor variations
        ("contractors obligation", "should normalize to Contractor's Obligations"),
        ("contractor's obligations", "should normalize to Contractor's Obligations"),
        ("Contractor's Obligation", "should normalize to Contractor's Obligations"),
        
        # Authority variations
        ("authoritys obligation", "should normalize to Authority's Obligations"),
        ("authority's obligations", "should normalize to Authority's Obligations"),
        
        # Issue types that shouldn't be categories
        ("completion certificate", "issue type → Others"),
        ("payment delay", "issue type → Others"),
        ("utility shifting", "issue type → Others"),
        
        # Unknown/invalid categories
        ("random category", "unknown → Others or rejected"),
        ("", "empty → rejected"),
    ]
    
    print("\nTesting individual category normalization:")
    print("-" * 80)
    
    for test_input, description in test_cases:
        result, status, confidence = normalizer.normalize_category(test_input)
        print(f"Input: '{test_input}' ({description})")
        print(f"  → Result: '{result}', Status: {status}, Confidence: {confidence:.2f}")
        print()
    
    # Reset stats for clean reporting
    normalizer.reset_stats()
    
    # Test with actual data file
    data_path = Path('data/raw/Consolidated_labeled_data.xlsx')
    if data_path.exists():
        print("\n" + "=" * 80)
        print("Validating actual training data")
        print("=" * 80)
        
        df = pd.read_excel(data_path)
        validation_results = normalizer.validate_data_quality(df)
        
        print(f"\nTotal rows analyzed: {validation_results['total_rows']}")
        print(f"Rows needing normalization: {len(validation_results['normalization_needed'])}")
        print(f"Rows with issues: {len(validation_results['rows_with_issues'])}")
        
        if validation_results['rows_with_issues']:
            print("\nSample of rows with issues (first 5):")
            for issue in validation_results['rows_with_issues'][:5]:
                print(f"  Row {issue['row']}: {issue['issue']}")
                print(f"    Issue Type: {issue['issue_type']}")
                print(f"    Category: {issue['category']}")
        
        print("\nCategory distribution after normalization:")
        for cat, count in sorted(validation_results['category_distribution'].items(), 
                                key=lambda x: x[1], reverse=True):
            print(f"  {cat:<30} {count:>4} samples")
        
        print(f"\nTotal unique categories: {len(validation_results['category_distribution'])}")
        
        # Export report
        report_path = './data/reports/category_normalization.json'
        report = normalizer.export_normalization_report(report_path)
        print(f"\nNormalization report exported to: {report_path}")
        
        # Show normalization statistics
        stats = normalizer.get_stats()
        print("\nNormalization Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Exact matches: {stats['exact_matches']}")
        print(f"  Normalized: {stats['normalized']}")
        print(f"  Fuzzy matched: {stats['fuzzy_matched']}")
        print(f"  Issue types as categories: {stats['issue_type_as_category']}")
        print(f"  Rejected: {stats['rejected']}")
    
    print("\n" + "=" * 80)
    print("Category Normalizer Test Complete")
    print("=" * 80)


def test_with_new_data():
    """Test normalizer with simulated new data."""
    print("\n" + "=" * 80)
    print("Testing with simulated new data ingestion")
    print("=" * 80)
    
    normalizer = CategoryNormalizer(strict_mode=True)
    
    # Simulate new data with various issues
    new_data = {
        'issue_type': [
            'New payment issue',
            'Contract amendment',
            'Scope change request',
            'Delay notification',
            'Completion certificate'  # This is an issue type
        ],
        'category': [
            'payment',  # Should normalize to Payments
            'cos',  # Should normalize to Change of Scope
            'Change of Scope, contractors obligation',  # Multiple categories
            'eot, DISPUTE RESOLUTION',  # Mixed case
            'completion certificate'  # Issue type used as category - should be flagged
        ]
    }
    
    df = pd.DataFrame(new_data)
    
    print("\nProcessing new data:")
    for idx, row in df.iterrows():
        print(f"\nRow {idx + 1}:")
        print(f"  Issue Type: {row['issue_type']}")
        print(f"  Original Category: {row['category']}")
        
        normalized = normalizer.parse_and_normalize_categories(row['category'])
        print(f"  Normalized Categories: {normalized}")
    
    # Validate the new data
    validation = normalizer.validate_data_quality(df)
    
    if validation['rows_with_issues']:
        print("\n⚠️  Issues detected in new data:")
        for issue in validation['rows_with_issues']:
            print(f"  Row {issue['row']}: {issue['issue']}")


if __name__ == "__main__":
    test_normalizer()
    test_with_new_data()