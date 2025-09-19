#!/usr/bin/env python3
"""
Test script for contract correspondence classifiers
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from classifier.hybrid_rag import HybridRAGClassifier


def setup_environment():
    """Load environment variables and configuration."""
    # Load .env file
    load_dotenv()
    
    # Load configuration
    config_manager = ConfigManager()
    
    # Validate configuration
    if not config_manager.validate_config():
        print("Configuration validation failed. Please check your config.yaml and .env files.")
        sys.exit(1)
    
    return config_manager


def initialize_components(config_manager):
    """Initialize core components."""
    # Check for synthetic training data first
    synthetic_training_path = './data/synthetic/combined_training_data.xlsx'
    original_training_path = './data/raw/Consolidated_labeled_data.xlsx'
    
    if Path(synthetic_training_path).exists():
        training_data_path = synthetic_training_path
        print(f"✅ Using synthetic + original training data ({training_data_path})")
    elif Path(original_training_path).exists():
        training_data_path = original_training_path
        print(f"Using original training data only ({training_data_path})")
    else:
        # Fallback to config
        data_paths = config_manager.get_data_paths()
        training_data_path = data_paths.get('training_data')
        
        if not training_data_path or not Path(training_data_path).exists():
            print(f"Training data not found. Expected locations:")
            print(f"  1. {synthetic_training_path} (synthetic + original)")
            print(f"  2. {original_training_path} (original only)")
            return None
    
    print(f"Loading training data from {training_data_path}...")
    
    # Initialize core components
    issue_mapper = IssueCategoryMapper(training_data_path)
    validator = ValidationEngine(training_data_path)
    data_analyzer = DataSufficiencyAnalyzer(training_data_path)
    
    print(f"Loaded {len(issue_mapper.get_all_issue_types())} issue types")
    print(f"Loaded {len(issue_mapper.get_all_categories())} categories")
    
    # Load validation data for evaluation
    validation_data_path = './data/synthetic/validation_set.xlsx'
    validation_data = None
    if Path(validation_data_path).exists():
        import pandas as pd
        validation_data = pd.read_excel(validation_data_path)
        print(f"Loaded {len(validation_data)} validation samples for evaluation")
    
    return {
        'issue_mapper': issue_mapper,
        'validator': validator,
        'data_analyzer': data_analyzer,
        'training_data_path': training_data_path,
        'validation_data': validation_data,
        'validation_data_path': validation_data_path
    }


def test_pure_llm_classifier(config_manager, components):
    """Test Pure LLM Classifier."""
    print("\n" + "="*60)
    print("Testing Pure LLM Classifier")
    print("="*60)
    
    # Check if Pure LLM is enabled
    if 'pure_llm' not in config_manager.get_enabled_approaches():
        print("Pure LLM approach is not enabled in config.yaml")
        return
    
    # Get configuration
    config = config_manager.get_approach_config('pure_llm')
    
    # Check for API key
    if not config.get('api_key'):
        print("API key not found. Please set OPENAI_API_KEY or CLAUDE_API_KEY in .env")
        return
    
    # Initialize classifier
    classifier = PureLLMClassifier(
        config=config,
        issue_mapper=components['issue_mapper'],
        validator=components['validator'],
        data_analyzer=components['data_analyzer']
    )
    
    # Test with sample text
    sample_text = """
    Subject: Request for Extension of Time - Milestone 3 Completion
    
    Dear Project Manager,
    
    We are writing to request an extension of time for the completion of Milestone 3
    due to unforeseen delays in material delivery. The original scope of work has also
    been modified as per the client's request, requiring additional resources and time.
    
    The delay in material delivery was caused by supply chain disruptions beyond our
    control. We request a 30-day extension to complete the work as per the revised
    specifications.
    
    Please review and approve this request at your earliest convenience.
    
    Best regards,
    Contractor Team
    """
    
    print("\nClassifying sample document...")
    print(f"Document preview: {sample_text[:200]}...")
    
    try:
        result = classifier.classify(sample_text)
        
        print("\n--- Classification Results ---")
        print(f"Status: Success")
        print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        print(f"Model Used: {result.get('model_used', 'N/A')}")
        
        print("\nIdentified Issues:")
        for issue in result.get('identified_issues', [])[:5]:
            print(f"  - {issue['issue_type']} (confidence: {issue['confidence']:.2f})")
        
        print("\nAssigned Categories:")
        for category in result.get('categories', [])[:5]:
            print(f"  - {category['category']} (confidence: {category['confidence']:.2f})")
        
        # Check for warnings
        warnings = result.get('data_sufficiency_warnings', [])
        if warnings:
            print("\n⚠️  Data Sufficiency Warnings:")
            for warning in warnings[:3]:
                print(f"  - {warning['message']}")
        
        # Check validation report
        validation = result.get('validation_report', {})
        if validation.get('hallucinations_detected'):
            print("\n⚠️  Hallucinations detected and corrected")
        
        # Save result for inspection
        output_file = Path('test_results_pure_llm.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during classification: {e}")


def test_hybrid_rag_classifier(config_manager, components):
    """Test Hybrid RAG+LLM Classifier."""
    print("\n" + "="*60)
    print("Testing Hybrid RAG+LLM Classifier")
    print("="*60)
    
    # Check if Hybrid RAG is enabled
    if 'hybrid_rag' not in config_manager.get_enabled_approaches():
        print("Hybrid RAG approach is not enabled in config.yaml")
        return
    
    # Get configuration
    config = config_manager.get_approach_config('hybrid_rag')
    
    # Initialize classifier
    classifier = HybridRAGClassifier(
        config=config,
        issue_mapper=components['issue_mapper'],
        validator=components['validator'],
        data_analyzer=components['data_analyzer']
    )
    
    # Build index if not exists
    index_path = Path('./data/embeddings/rag_index')
    if not index_path.with_suffix('.faiss').exists():
        print("\nBuilding vector index from training data...")
        print("This may take a few minutes on first run...")
        classifier.build_index(
            components['training_data_path'],
            save_path=str(index_path)
        )
        print("Index built and saved successfully!")
    else:
        print("\nUsing existing vector index")
    
    # Get index statistics
    stats = classifier.get_index_stats()
    print(f"Index contains {stats.get('total_vectors', 0)} vectors")
    
    # Test with sample text
    sample_text = """
    Subject: Payment Delay Notification - Invoice #2024-001
    
    Dear Contractor,
    
    We regret to inform you that the payment for Invoice #2024-001 will be delayed
    due to pending approval from the finance department. The invoice amount of
    $250,000 for completed work on Section A is currently under review.
    
    Additionally, we have identified some discrepancies in the bill of quantities
    that need to be resolved before processing the payment. Please provide
    clarification on the items listed in Appendix B.
    
    We expect the payment to be processed within 15 business days after receiving
    your clarification.
    
    Regards,
    Project Finance Team
    """
    
    print("\nClassifying sample document...")
    print(f"Document preview: {sample_text[:200]}...")
    
    try:
        result = classifier.classify(sample_text)
        
        print("\n--- Classification Results ---")
        print(f"Status: Success")
        print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        print(f"Search Results Used: {result.get('search_results_used', 0)}")
        
        print("\nIdentified Issues:")
        for issue in result.get('identified_issues', [])[:5]:
            source = issue.get('source', 'unknown')
            print(f"  - {issue['issue_type']} (confidence: {issue['confidence']:.2f}, source: {source})")
        
        print("\nAssigned Categories:")
        for category in result.get('categories', [])[:5]:
            print(f"  - {category['category']} (confidence: {category['confidence']:.2f})")
            if category.get('source_issues'):
                for source in category['source_issues'][:2]:
                    print(f"    ← from: {source['issue_type']}")
        
        # Check for warnings
        warnings = result.get('data_sufficiency_warnings', [])
        if warnings:
            print("\n⚠️  Data Sufficiency Warnings:")
            for warning in warnings[:3]:
                print(f"  - {warning['message']}")
        
        # Save result for inspection
        output_file = Path('test_results_hybrid_rag.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nFull results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during classification: {e}")


def generate_data_sufficiency_report(components):
    """Generate data sufficiency analysis report."""
    print("\n" + "="*60)
    print("Data Sufficiency Analysis")
    print("="*60)
    
    analyzer = components['data_analyzer']
    report = analyzer.generate_sufficiency_report()
    
    print(f"\nTotal Training Samples: {report['summary']['total_samples']}")
    print(f"Unique Issue Types: {report['summary']['unique_issue_types']}")
    print(f"Unique Categories: {report['summary']['unique_categories']}")
    
    # Show critical items
    if report['critical_issues']:
        print(f"\n⚠️  CRITICAL: {len(report['critical_issues'])} issue types with <5 samples:")
        for item in report['critical_issues'][:5]:
            print(f"  - {item['issue_type']}: {item['sample_count']} samples")
    
    if report['critical_categories']:
        print(f"\n⚠️  CRITICAL: {len(report['critical_categories'])} categories with <5 samples:")
        for item in report['critical_categories'][:5]:
            print(f"  - {item['category']}: {item['sample_count']} samples")
    
    # Show recommendations
    if report['recommendations']:
        print("\n📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"  [{rec['priority']}] {rec['message']}")
    
    # Save full report
    output_file = Path('data_sufficiency_report.json')
    analyzer.save_report(report, str(output_file))
    print(f"\nFull report saved to {output_file}")


def run_validation_evaluation(config_manager, components):
    """Run evaluation on validation dataset."""
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    
    validation_data = components.get('validation_data')
    if validation_data is None:
        print("No validation data available. Skipping evaluation.")
        return
    
    print(f"Evaluating on {len(validation_data)} validation samples...")
    
    # Initialize metrics tracking
    evaluation_results = {
        'total_samples': len(validation_data),
        'real_samples': len(validation_data[validation_data.get('is_synthetic', False) == False]),
        'synthetic_samples': len(validation_data[validation_data.get('is_synthetic', False) == True]),
        'approaches': {}
    }
    
    # Test enabled approaches
    enabled_approaches = config_manager.get_enabled_approaches()
    
    # Test samples (first 10 for quick evaluation)
    sample_size = min(10, len(validation_data))
    test_samples = validation_data.head(sample_size)
    
    print(f"Running evaluation on {sample_size} samples...")
    
    for approach in enabled_approaches:
        print(f"\n--- Evaluating {approach.upper()} approach ---")
        
        try:
            # Initialize classifier
            if approach == 'pure_llm':
                config = config_manager.get_approach_config('pure_llm')
                if not config.get('api_key'):
                    print(f"Skipping {approach} - no API key configured")
                    continue
                    
                classifier = PureLLMClassifier(
                    config=config,
                    issue_mapper=components['issue_mapper'],
                    validator=components['validator'],
                    data_analyzer=components['data_analyzer']
                )
            
            elif approach == 'hybrid_rag':
                config = config_manager.get_approach_config('hybrid_rag')
                classifier = HybridRAGClassifier(
                    config=config,
                    issue_mapper=components['issue_mapper'],
                    validator=components['validator'],
                    data_analyzer=components['data_analyzer']
                )
                
                # Ensure index exists
                index_path = Path('./data/embeddings/rag_index')
                if not index_path.with_suffix('.faiss').exists():
                    print(f"Building index for {approach}...")
                    classifier.build_index(
                        components['training_data_path'],
                        save_path=str(index_path)
                    )
            
            # Run evaluation
            approach_results = {
                'correct_predictions': 0,
                'total_predictions': 0,
                'processing_times': [],
                'sample_results': []
            }
            
            for idx, row in test_samples.iterrows():
                # Create test text
                subject = str(row.get('subject', ''))
                body = str(row.get('body', ''))
                test_text = f"Subject: {subject}\n\nBody: {body}" if body != 'nan' else subject
                
                # Get expected results
                expected_issue = str(row.get('issue_type', ''))
                expected_category = str(row.get('category', ''))
                
                try:
                    # Classify
                    result = classifier.classify(test_text)
                    
                    # Extract predictions
                    predicted_issues = [issue['issue_type'] for issue in result.get('identified_issues', [])]
                    predicted_categories = [cat['category'] for cat in result.get('categories', [])]
                    
                    # Check accuracy (simple exact match for now)
                    issue_correct = expected_issue in predicted_issues
                    category_correct = expected_category in predicted_categories
                    
                    if issue_correct and category_correct:
                        approach_results['correct_predictions'] += 1
                    
                    approach_results['total_predictions'] += 1
                    approach_results['processing_times'].append(result.get('processing_time', 0))
                    
                    approach_results['sample_results'].append({
                        'expected_issue': expected_issue,
                        'predicted_issues': predicted_issues[:3],
                        'expected_category': expected_category,
                        'predicted_categories': predicted_categories[:3],
                        'issue_correct': issue_correct,
                        'category_correct': category_correct,
                        'is_synthetic': row.get('is_synthetic', False)
                    })
                    
                    print(f"  Sample {idx+1}: {'✓' if (issue_correct and category_correct) else '✗'}")
                    
                except Exception as e:
                    print(f"  Sample {idx+1}: Error - {str(e)[:100]}...")
                    approach_results['sample_results'].append({
                        'error': str(e),
                        'expected_issue': expected_issue,
                        'expected_category': expected_category
                    })
            
            # Calculate metrics
            if approach_results['total_predictions'] > 0:
                accuracy = approach_results['correct_predictions'] / approach_results['total_predictions']
                avg_time = sum(approach_results['processing_times']) / len(approach_results['processing_times']) if approach_results['processing_times'] else 0
                
                print(f"\n{approach.upper()} Results:")
                print(f"  Accuracy: {accuracy:.2%} ({approach_results['correct_predictions']}/{approach_results['total_predictions']})")
                print(f"  Avg Processing Time: {avg_time:.2f}s")
                
                evaluation_results['approaches'][approach] = {
                    'accuracy': accuracy,
                    'correct_predictions': approach_results['correct_predictions'],
                    'total_predictions': approach_results['total_predictions'],
                    'avg_processing_time': avg_time,
                    'sample_results': approach_results['sample_results']
                }
            
        except Exception as e:
            print(f"Error evaluating {approach}: {e}")
    
    # Save evaluation results
    output_file = Path('validation_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"\nEvaluation results saved to {output_file}")
    
    return evaluation_results


def main():
    """Main test function."""
    print("Contract Correspondence Classification System - Test Suite")
    print("="*60)
    
    # Setup environment
    config_manager = setup_environment()
    
    # Initialize components
    components = initialize_components(config_manager)
    if not components:
        return
    
    # Generate data sufficiency report
    generate_data_sufficiency_report(components)
    
    # Run validation evaluation
    run_validation_evaluation(config_manager, components)
    
    # Test classifiers with sample data
    enabled_approaches = config_manager.get_enabled_approaches()
    print(f"\nEnabled approaches: {enabled_approaches}")
    
    if 'pure_llm' in enabled_approaches:
        test_pure_llm_classifier(config_manager, components)
    
    if 'hybrid_rag' in enabled_approaches:
        test_hybrid_rag_classifier(config_manager, components)
    
    if not enabled_approaches:
        print("\nNo classification approaches are enabled.")
        print("Please enable at least one approach in config.yaml")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()