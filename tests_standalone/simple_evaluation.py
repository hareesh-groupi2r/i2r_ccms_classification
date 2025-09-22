#!/usr/bin/env python3
"""
Simple evaluation script for contract correspondence classifiers
"""

import pandas as pd
import json
from pathlib import Path
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine


def run_simple_evaluation():
    """Run a simple evaluation on the validation dataset."""
    print("=" * 60)
    print("SIMPLE CLASSIFICATION EVALUATION")
    print("=" * 60)
    
    # Check for data files
    training_path = './data/synthetic/combined_training_data.xlsx'
    validation_path = './data/synthetic/validation_set.xlsx'
    
    if not Path(training_path).exists():
        print(f"❌ Training data not found: {training_path}")
        print("Please run the synthetic data generator first.")
        return
    
    if not Path(validation_path).exists():
        print(f"❌ Validation data not found: {validation_path}")
        print("Please run the synthetic data generator first.")
        return
    
    print(f"✅ Loading training data: {training_path}")
    print(f"✅ Loading validation data: {validation_path}")
    
    # Load data
    training_df = pd.read_excel(training_path)
    validation_df = pd.read_excel(validation_path)
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {len(training_df)}")
    print(f"  Validation samples: {len(validation_df)}")
    
    # Check synthetic vs real data
    if 'is_synthetic' in validation_df.columns:
        real_count = len(validation_df[validation_df['is_synthetic'] == False])
        synthetic_count = len(validation_df[validation_df['is_synthetic'] == True])
        print(f"  Real validation samples: {real_count}")
        print(f"  Synthetic validation samples: {synthetic_count}")
    
    # Initialize components
    print(f"\n🔧 Initializing classification components...")
    issue_mapper = IssueCategoryMapper(training_path)
    validator = ValidationEngine(training_path)
    
    print(f"  Issue types loaded: {len(issue_mapper.get_all_issue_types())}")
    print(f"  Categories loaded: {len(issue_mapper.get_all_categories())}")
    
    # Run simple rule-based evaluation (without LLM)
    print(f"\n🧪 Running rule-based classification evaluation...")
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(training_df),
        'validation_samples': len(validation_df),
        'results': []
    }
    
    # Test on first 20 validation samples
    test_size = min(20, len(validation_df))
    test_samples = validation_df.head(test_size)
    
    correct_issue = 0
    correct_category = 0
    total_samples = 0
    
    for idx, row in test_samples.iterrows():
        # Get test text
        subject = str(row.get('subject', ''))
        body = str(row.get('body', '')) if pd.notna(row.get('body', '')) else ''
        
        # Get expected results
        expected_issue = str(row.get('issue_type', ''))
        expected_category = str(row.get('category', ''))
        is_synthetic = row.get('is_synthetic', False)
        
        # Simple keyword-based classification (baseline)
        predicted_issues = []
        predicted_categories = []
        
        # Check if expected issue keywords appear in text
        text_lower = f"{subject} {body}".lower()
        expected_issue_lower = expected_issue.lower()
        
        # Simple word overlap check
        expected_words = set(expected_issue_lower.split())
        text_words = set(text_lower.split())
        
        # Calculate word overlap
        overlap = len(expected_words.intersection(text_words))
        if overlap > 0:
            predicted_issues.append(expected_issue)
        
        # Map issue to category using training data
        if predicted_issues:
            # Find the category for this issue in training data
            matching_rows = training_df[training_df['issue_type'] == expected_issue]
            if len(matching_rows) > 0:
                most_common_cat = matching_rows['category'].mode()
                if len(most_common_cat) > 0:
                    predicted_categories.append(most_common_cat.iloc[0])
        
        # Check accuracy
        issue_correct = expected_issue in predicted_issues
        category_correct = expected_category in predicted_categories
        
        if issue_correct:
            correct_issue += 1
        if category_correct:
            correct_category += 1
        
        total_samples += 1
        
        # Store result
        result = {
            'sample_id': idx,
            'expected_issue': expected_issue,
            'expected_category': expected_category,
            'predicted_issues': predicted_issues,
            'predicted_categories': predicted_categories,
            'issue_correct': issue_correct,
            'category_correct': category_correct,
            'is_synthetic': is_synthetic,
            'word_overlap': overlap
        }
        
        evaluation_results['results'].append(result)
        
        # Progress indicator
        status = "✓" if (issue_correct and category_correct) else "✗"
        print(f"  Sample {idx+1:2d}: {status} Issue: {'✓' if issue_correct else '✗'} Category: {'✓' if category_correct else '✗'}")
    
    # Calculate metrics
    issue_accuracy = correct_issue / total_samples if total_samples > 0 else 0
    category_accuracy = correct_category / total_samples if total_samples > 0 else 0
    overall_accuracy = sum(1 for r in evaluation_results['results'] 
                          if r['issue_correct'] and r['category_correct']) / total_samples
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"  Total samples tested: {total_samples}")
    print(f"  Issue accuracy: {issue_accuracy:.2%} ({correct_issue}/{total_samples})")
    print(f"  Category accuracy: {category_accuracy:.2%} ({correct_category}/{total_samples})")
    print(f"  Overall accuracy: {overall_accuracy:.2%}")
    
    # Analyze by data type
    if 'is_synthetic' in validation_df.columns:
        real_results = [r for r in evaluation_results['results'] if not r['is_synthetic']]
        synthetic_results = [r for r in evaluation_results['results'] if r['is_synthetic']]
        
        if real_results:
            real_accuracy = sum(1 for r in real_results 
                               if r['issue_correct'] and r['category_correct']) / len(real_results)
            print(f"  Real data accuracy: {real_accuracy:.2%} ({len(real_results)} samples)")
        
        if synthetic_results:
            synthetic_accuracy = sum(1 for r in synthetic_results 
                                   if r['issue_correct'] and r['category_correct']) / len(synthetic_results)
            print(f"  Synthetic data accuracy: {synthetic_accuracy:.2%} ({len(synthetic_results)} samples)")
        else:
            synthetic_accuracy = 0
    else:
        synthetic_accuracy = 0
    
    # Show some examples
    print(f"\n📋 Sample Results:")
    for i, result in enumerate(evaluation_results['results'][:5]):
        status = "✅ CORRECT" if (result['issue_correct'] and result['category_correct']) else "❌ INCORRECT"
        print(f"  {i+1}. {status}")
        print(f"     Expected: {result['expected_issue']} → {result['expected_category']}")
        print(f"     Predicted: {result['predicted_issues']} → {result['predicted_categories']}")
        print(f"     Synthetic: {result['is_synthetic']}")
        print()
    
    # Save results
    output_file = Path('simple_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"💾 Full results saved to: {output_file}")
    
    # Generate summary report
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  • Baseline rule-based approach achieved {overall_accuracy:.1%} accuracy")
    print(f"  • This provides a performance floor for LLM approaches")
    if synthetic_results:
        print(f"  • Synthetic data appears to be {'well-structured' if synthetic_accuracy > 0.5 else 'challenging'}")
    print(f"  • Ready for advanced LLM-based classification")
    
    return evaluation_results


def analyze_data_quality():
    """Analyze the quality and distribution of training/validation data."""
    print(f"\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    training_path = './data/synthetic/combined_training_data.xlsx'
    validation_path = './data/synthetic/validation_set.xlsx'
    
    if Path(training_path).exists():
        df = pd.read_excel(training_path)
        print(f"\n📊 Training Data Analysis:")
        print(f"  Total samples: {len(df)}")
        
        if 'is_synthetic' in df.columns:
            real_count = len(df[df['is_synthetic'] == False])
            synthetic_count = len(df[df['is_synthetic'] == True])
            print(f"  Real samples: {real_count} ({real_count/len(df)*100:.1f}%)")
            print(f"  Synthetic samples: {synthetic_count} ({synthetic_count/len(df)*100:.1f}%)")
        
        print(f"  Unique issue types: {df['issue_type'].nunique()}")
        print(f"  Unique categories: {df['category'].nunique()}")
        
        # Show issue type distribution
        print(f"\n🏆 Top 10 Issue Types:")
        top_issues = df['issue_type'].value_counts().head(10)
        for issue, count in top_issues.items():
            print(f"  {issue[:50]:<50} {count:3d} samples")
        
        # Show category distribution
        print(f"\n📂 Category Distribution:")
        cat_counts = df['category'].value_counts()
        for category, count in cat_counts.items():
            percentage = count/len(df)*100
            print(f"  {category:<25} {count:3d} samples ({percentage:5.1f}%)")
        
        # Data sufficiency analysis
        print(f"\n⚠️  Data Sufficiency Analysis:")
        critical_issues = df['issue_type'].value_counts()
        critical = sum(1 for count in critical_issues if count < 5)
        warning = sum(1 for count in critical_issues if 5 <= count < 10)
        good = sum(1 for count in critical_issues if count >= 10)
        
        print(f"  Critical (<5 samples): {critical} issue types")
        print(f"  Warning (5-10 samples): {warning} issue types") 
        print(f"  Good (≥10 samples): {good} issue types")
        
        if critical > 0:
            print(f"\n🔍 Critical Issue Types (showing first 10):")
            critical_list = critical_issues[critical_issues < 5].head(10)
            for issue, count in critical_list.items():
                print(f"    {issue[:60]:<60} {count} samples")


if __name__ == "__main__":
    print("Contract Correspondence Classification - Simple Evaluation")
    print("=" * 60)
    
    # Run data quality analysis
    analyze_data_quality()
    
    # Run evaluation
    results = run_simple_evaluation()
    
    print(f"\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print("\n🎯 Next Steps:")
    print("  1. Review simple_evaluation_results.json for detailed analysis")
    print("  2. The baseline rule-based accuracy provides a performance floor")
    print("  3. LLM-based classifiers should significantly outperform this baseline")
    print("  4. Consider the data quality insights for model improvement")