#!/usr/bin/env python3
"""
Demonstration of the Category Normalization System
Shows how the system handles various category inputs and maintains data quality
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.category_normalizer import CategoryNormalizer
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer


def main():
    print("=" * 80)
    print("CONTRACT CORRESPONDENCE CLASSIFICATION SYSTEM")
    print("Category Normalization & Data Quality Demonstration")
    print("=" * 80)
    
    # Initialize components
    training_path = 'data/raw/Consolidated_labeled_data.xlsx'
    mapper = IssueCategoryMapper(training_path)
    validator = ValidationEngine(training_path)
    analyzer = DataSufficiencyAnalyzer(training_path)
    normalizer = CategoryNormalizer(strict_mode=False)
    
    # Display the 8 standard categories
    print("\n📋 STANDARD CATEGORIES (Only 8 allowed):")
    print("-" * 40)
    for i, cat in enumerate(CategoryNormalizer.STANDARD_CATEGORIES, 1):
        count = analyzer.category_counts.get(cat, 0)
        print(f"{i}. {cat:<30} ({count} training samples)")
    
    # Show how various inputs are normalized
    print("\n🔄 CATEGORY NORMALIZATION EXAMPLES:")
    print("-" * 40)
    
    test_inputs = [
        ("eot", "Common lowercase variation"),
        ("EOT", "All caps variation"),
        ("cos", "Abbreviation for Change of Scope"),
        ("payment", "Singular form of Payments"),
        ("contractors obligation", "Missing apostrophe"),
        ("completion certificate", "Issue type incorrectly used as category"),
        ("random text", "Unknown category"),
    ]
    
    for input_cat, description in test_inputs:
        normalized, status, confidence = normalizer.normalize_category(input_cat)
        print(f"Input: '{input_cat}' ({description})")
        print(f"  → Normalized to: '{normalized}' (status: {status}, confidence: {confidence:.2f})")
    
    # Simulate ingesting new data with various issues
    print("\n📥 SIMULATING NEW DATA INGESTION:")
    print("-" * 40)
    
    # Create sample new data with various category issues
    new_data = pd.DataFrame({
        'issue_type': [
            'New Contract Issue',
            'Payment Delay Notice',
            'Scope Modification Request',
            'Authority Decision Pending',
            'Force Majeure Event'
        ],
        'category': [
            'payment, eot',  # Multiple categories with variations
            'PAYMENTS',  # All caps
            'cos, contractors obligation',  # Abbreviation and missing apostrophe
            'Authority\'s Obligation, dispute resolution',  # Mixed case
            'completion certificate, others'  # Issue type as category
        ],
        'subject': [
            'Regarding payment and timeline',
            'Notice of payment delay',
            'Request for scope changes',
            'Awaiting authority decision',
            'Force majeure notification'
        ]
    })
    
    print("Processing new data batch...")
    print()
    
    for idx, row in new_data.iterrows():
        print(f"Record {idx + 1}:")
        print(f"  Issue Type: {row['issue_type']}")
        print(f"  Original Categories: {row['category']}")
        
        # Normalize categories
        normalized_cats = normalizer.parse_and_normalize_categories(row['category'])
        print(f"  ✓ Normalized to: {', '.join(normalized_cats)}")
        
        # Validate using the validation engine
        for cat in row['category'].split(','):
            cat = cat.strip()
            validated_cat, is_valid, conf = validator.validate_category(cat)
            if not is_valid and validated_cat:
                print(f"    ⚠️ '{cat}' was corrected to '{validated_cat}'")
            elif not validated_cat:
                print(f"    ❌ '{cat}' was rejected")
        print()
    
    # Data quality validation
    print("\n📊 DATA QUALITY VALIDATION:")
    print("-" * 40)
    
    validation_results = normalizer.validate_data_quality(new_data)
    
    print(f"Total rows processed: {validation_results['total_rows']}")
    print(f"Rows needing normalization: {len(validation_results['normalization_needed'])}")
    print(f"Rows with issues: {len(validation_results['rows_with_issues'])}")
    
    if validation_results['rows_with_issues']:
        print("\nIssues detected:")
        for issue in validation_results['rows_with_issues']:
            print(f"  • Row {issue['row']}: {issue['issue']}")
    
    # Show statistics
    print("\n📈 NORMALIZATION STATISTICS:")
    print("-" * 40)
    stats = normalizer.get_stats()
    print(f"Total categories processed: {stats['total_processed']}")
    print(f"Exact matches: {stats['exact_matches']}")
    print(f"Normalized (variations fixed): {stats['normalized']}")
    print(f"Fuzzy matched: {stats['fuzzy_matched']}")
    print(f"Issue types incorrectly used as categories: {stats['issue_type_as_category']}")
    print(f"Rejected (invalid): {stats['rejected']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SYSTEM READY FOR PRODUCTION")
    print("=" * 80)
    print("\n🎯 Key Features:")
    print("  • Enforces exactly 8 standard categories")
    print("  • Automatically normalizes common variations")
    print("  • Detects when issue types are used as categories")
    print("  • Provides confidence scores for all normalizations")
    print("  • Maintains data quality across all ingested data")
    print("\n💾 The normalization mappings are saved and will be consistent")
    print("   for all future data ingestion, ensuring data quality.")


if __name__ == "__main__":
    main()