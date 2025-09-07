#!/usr/bin/env python3
"""
Test script for Issue Type Normalizer
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.issue_normalizer import IssueTypeNormalizer


def main():
    print("=" * 80)
    print("Testing Issue Type Normalizer")
    print("=" * 80)
    
    # Initialize normalizer
    normalizer = IssueTypeNormalizer()
    
    # Test cases from the analysis
    test_cases = [
        # Case differences
        ("Utility Shifting", "Utility shifting", "Should normalize to lowercase version"),
        ("Progress Review", "Progress review", "Should keep the more common version"),
        
        # Punctuation differences
        ("Safety measures during construction ", "Safety measures during construction.", "Should normalize punctuation"),
        ("Borrow area ", "Borrow area", "Should remove trailing space"),
        
        # Typos
        ("Delay in construction activiites", "Delay in construction activities", "Should fix typo"),
        ("Under uitilisation / idling of resources", "Under utilisation / idling of resources", "Should fix typo"),
        
        # Singular/plural
        ("Extension of time proposal", "Extension of Time Proposals", "Should normalize to plural"),
        ("Change of scope proposals", "Change of scope proposal", "Should normalize to singular"),
        
        # Complex variations
        ("Handing over of land /Possession of site", "Handing over of land /Possession of site.  ", "Should normalize spacing/punctuation"),
    ]
    
    print("\nTesting individual issue type normalizations:")
    print("-" * 80)
    
    for original, expected, description in test_cases:
        result, status, confidence = normalizer.normalize_issue_type(original)
        print(f"Test: {description}")
        print(f"  Input: '{original}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{result}' (status: {status}, confidence: {confidence:.2f})")
        
        # Check if normalization worked as expected
        if result == expected:
            print("  ✅ PASS")
        else:
            print("  ❌ Different result than expected")
        print()
    
    # Test with actual data
    data_path = Path('data/raw/Consolidated_labeled_data.xlsx')
    if data_path.exists():
        print("\n" + "=" * 80)
        print("Testing with actual training data")
        print("=" * 80)
        
        df = pd.read_excel(data_path)
        validation_results = normalizer.validate_data_quality(df)
        
        print(f"\\nOriginal unique issue types: {validation_results['statistics']['original_unique_count']}")
        print(f"Normalized unique issue types: {validation_results['statistics']['normalized_unique_count']}")
        print(f"Reduction: {validation_results['statistics']['reduction']} issue types")
        print(f"Normalization rate: {validation_results['statistics']['normalization_rate']:.1%}")
        
        if validation_results['potential_duplicates']:
            print(f"\\nFound {len(validation_results['potential_duplicates'])} groups that would be merged:")
            print("-" * 60)
            
            for i, duplicate in enumerate(validation_results['potential_duplicates'][:10], 1):  # Show first 10
                print(f"\\n{i}. Would normalize to: '{duplicate['normalized']}'")
                print(f"   Total samples: {duplicate['total_samples']}")
                for original, count in duplicate['sample_counts'].items():
                    print(f"   - '{original}' ({count} samples)")
            
            if len(validation_results['potential_duplicates']) > 10:
                print(f"   ... and {len(validation_results['potential_duplicates']) - 10} more groups")
        
        # Show normalization statistics
        stats = normalizer.get_stats()
        print("\\nNormalization Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Exact matches: {stats['exact_matches']}")
        print(f"  Normalized: {stats['normalized']}")
        print(f"  Fuzzy matched: {stats['fuzzy_matched']}")
        print(f"  Rejected: {stats['rejected']}")
        
        # Export report
        report_path = './data/reports/issue_normalization.json'
        normalizer.export_normalization_report(report_path)
        print(f"\\nNormalization report exported to: {report_path}")
    
    print("\\n" + "=" * 80)
    print("Issue Type Normalizer Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()